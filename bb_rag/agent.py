"""Agent mit Werkzeugen (LlamaIndex FunctionAgent + Llama 3.1 über Ollama, lokal).

Das Sprachmodell entscheidet selbst, welche Werkzeuge es aufruft:
  semantische Suche (Vektordatenbank) · Volltextsuche · Ortsregister · Jahrgangsvergleich.
Jeder Werkzeugaufruf wird protokolliert, damit nachvollziehbar bleibt, worauf eine Antwort beruht.

Aufruf:  python -m bb_rag.agent "Wie oft wird Tallinn im Vergleich zu Reval genannt?"
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from llama_index.core.agent.workflow import AgentStream, FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from . import config, tools

SYSTEM = (
    "Du bist ein Rechercheassistent für die Zeitschrift „Baltische Briefe“ (deutschbaltische Diaspora). "
    "Das Korpus umfasst 48 Ausgaben aus den Jahrgängen 1949, 1950, 1988, 1991 und 2004. "
    "Nutze die Werkzeuge, bevor du antwortest, und erfinde nichts: "
    "inhaltliche Fragen → semantic_search; ein bestimmtes Wort oder ein Name → fulltext_search; "
    "Orte, Schreibformen, Koordinaten → place_register; Entwicklung über die Jahrgänge → compare_years. "
    "Antworte auf Deutsch, knapp, und nenne die Belege (Abschnitts- oder Ausgaben-IDs wie bb_04_49#003). "
    "Weise darauf hin, dass Texte aus OCR stammen und Orte automatisch, ungeprüft zugeordnet sind. "
    "Bei Rückfragen („und bei …?“) beziehe dich auf das bisherige Gespräch und nutze erneut die Werkzeuge. "
    "Bei Vergleichen mehrerer Wörter rufe compare_years EINMAL mit allen Wörtern auf, z. B. \"Reval, Tallinn\". "
    "Tabellen aus compare_years werden automatisch unter deine Antwort gesetzt: Schreibe dann keine Zahlen aus der "
    "Tabelle ab und rechne nichts aus, sondern leite nur kurz ein. Andere Zahlen nennst du nur wörtlich so, wie sie im "
    "Werkzeugergebnis stehen, mit der genauen Schreibform (z. B. „Dorpat/Tartu“ ist nicht „Dorpat“). Meldet ein Werkzeug einen Fehler, "
    "rufe es korrigiert erneut auf, statt Zahlen zu schätzen. Lass optionale Parameter weg, wenn sie nicht gebraucht werden."
)


def semantic_search(query: str, year: int | None = None, place: str | None = None) -> str:
    """Semantische Suche nach inhaltlich passenden Textabschnitten.

    Args:
        query: Frage oder Thema in natürlicher Sprache, z. B. "Hilfe für Flüchtlinge".
        year: optional ein Jahrgang (1949, 1950, 1988, 1991 oder 2004).
        place: optional ein Ort, der im Abschnitt erwähnt sein muss, z. B. "Riga".
    """
    from .search import search  # lädt Embedding-Modell und Index erst bei Bedarf
    try:
        year = tools.normalize_year(year)
    except ValueError as err:
        return str(err)
    place = None if place is None or str(place).strip().lower() in ("", "none", "null") else str(place).strip()
    hits = search(query, year=year, place=place, top_k=5)
    if not hits:
        return "Keine passenden Abschnitte gefunden."
    return "\n\n".join(f"[{h.chunk_id}] {h.title} (Ähnlichkeit {h.score}):\n{h.text[:700]}" for h in hits)


def fulltext_search(term: str, year: int | None = None) -> str:
    """Exakte Wortsuche im Volltext mit Trefferzahlen pro Jahrgang und Textstellen.

    Args:
        term: gesuchtes Wort oder Name in genauer Schreibweise, z. B. "Dorpat".
        year: optional ein Jahrgang (1949, 1950, 1988, 1991 oder 2004).
    """
    return tools.fulltext_search(term, year=year)


def place_register(name: str) -> str:
    """Ortsregister: alle Schreibformen eines Ortes mit Nennungen je Jahrgang, Prüfstatus und Koordinate.

    Args:
        name: Ortsname oder Teil davon, z. B. "Reval" oder "Tartu".
    """
    return tools.place_register(name)


def compare_years(terms: str) -> str:
    """Vergleicht, wie häufig ein oder mehrere Wörter in den fünf Jahrgängen vorkommen; liefert eine Tabelle.

    Args:
        terms: ein Wort oder mehrere, durch Komma getrennt, z. B. "Reval, Tallinn".
    """
    return tools.compare_years(terms)


TOOLS = [FunctionTool.from_defaults(fn) for fn in (semantic_search, fulltext_search, place_register, compare_years)]


@dataclass
class AgentAnswer:
    question: str
    text: str
    steps: list[tuple[str, dict, str]] = field(default_factory=list)   # (Werkzeug, Argumente, Ergebnis)


def build_agent() -> FunctionAgent:
    llm = Ollama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL, request_timeout=300.0,
                 temperature=0.0, context_window=8192, is_function_calling_model=True)
    return FunctionAgent(tools=TOOLS, llm=llm, system_prompt=SYSTEM)


MAX_TURNS = 6   # so viele frühere Nachrichten bleiben im Gedächtnis (kleines Kontextfenster des 8B-Modells)


def history_messages(history: list[tuple[str, str]] | None) -> list[ChatMessage]:
    """Frühere Gesprächsschritte als ChatMessages; Werkzeugprotokolle (<details>…) werden entfernt."""
    out = []
    for role, content in (history or [])[-MAX_TURNS:]:
        if role not in ("user", "assistant") or not content:
            continue
        text = re.sub(r"<details>.*?</details>", "", str(content), flags=re.S).strip()
        out.append(ChatMessage(role=role, content=text[:1500]))
    return out


TOOL_LABELS = {"semantic_search": "durchsucht die Texte inhaltlich", "fulltext_search": "sucht im Volltext",
               "place_register": "schlägt im Ortsregister nach", "compare_years": "vergleicht die Jahrgänge"}


async def stream(question: str, history: list[tuple[str, str]] | None = None) -> AsyncIterator[dict]:
    """Ereignisse während der Bearbeitung, für eine Live-Anzeige:

    {"type": "tool_call", "name", "kwargs"} · {"type": "tool_result", "name", "kwargs", "output"} ·
    {"type": "delta", "text"} (Antworttext Stück für Stück) · {"type": "done", "answer": AgentAnswer}
    """
    handler = build_agent().run(user_msg=question, chat_history=history_messages(history) or None)
    steps = []
    async for event in handler.stream_events():
        if isinstance(event, ToolCall) and not isinstance(event, ToolCallResult):
            yield {"type": "tool_call", "name": event.tool_name, "kwargs": dict(event.tool_kwargs)}
        elif isinstance(event, ToolCallResult):
            steps.append((event.tool_name, dict(event.tool_kwargs), str(event.tool_output)))
            yield {"type": "tool_result", "name": event.tool_name, "kwargs": dict(event.tool_kwargs),
                   "output": str(event.tool_output)}
        elif isinstance(event, AgentStream) and event.delta:
            yield {"type": "delta", "text": event.delta}
    response = await handler
    text = check_citations(strip_tool_json(str(response).strip()), steps)
    yield {"type": "done", "answer": AgentAnswer(question, attach_tables(text, steps), steps)}


async def _run(question: str, history: list[tuple[str, str]] | None = None) -> AgentAnswer:
    async for event in stream(question, history):
        if event["type"] == "done":
            return event["answer"]
    raise RuntimeError("Agent lieferte keine Antwort")


TOOL_JSON = re.compile(r'\{\s*"name"\s*:\s*"\w+"\s*,\s*"(?:parameters|arguments)"\s*:\s*\{.*?\}\s*\}', re.S)
CITATION = re.compile(r"\bbb_\d+_\d+(?:#\d+)?\b")


def strip_tool_json(text: str) -> str:
    """Llama 3.1 schreibt manchmal einen Werkzeugaufruf als rohes JSON in den Antworttext – entfernen."""
    return re.sub(r"\n{3,}", "\n\n", TOOL_JSON.sub("", text)).strip()


def check_citations(text: str, steps: list[tuple[str, dict, str]]) -> str:
    """Belege prüfen: Eine Abschnitts-ID (bb_11_49#013) muss in einem Werkzeugergebnis vorkommen.

    Erfundene Abschnittsnummern werden auf die Ausgabe gekürzt, wenn diese belegt ist (bb_04_49#003 → bb_04_49),
    sonst als ungeprüft markiert.
    """
    sources = "\n".join(out for _, _, out in steps)

    def fix(match: re.Match) -> str:
        cid = match.group(0)
        if cid in sources:
            return cid
        doc = cid.split("#")[0]
        if doc in sources:
            return doc
        return f"{cid} (Beleg nicht gefunden)"

    return CITATION.sub(fix, text)


COMPARATIVE = re.compile(r"häufiger|häufigst|seltener|öfter|mehr als|weniger als|überwieg|in allen|am meisten|"
                         r"am häufigsten|doppelt|Mal so|gleich oft", re.IGNORECASE)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def attach_tables(text: str, steps: list[tuple[str, dict, str]]) -> str:
    """Tabellen und berechnete Kurzfassungen aus Werkzeugergebnissen deterministisch an die Antwort hängen.

    Kleine Modelle verweisen oft auf „die Tabelle“, ohne sie wiederzugeben, oder deuten Zahlen falsch
    („Tartu wird häufiger genannt“, obwohl die Tabelle 60 zu 132 für Dorpat zeigt). Deshalb:
    - Tabelle und Kurzfassung kommen direkt aus dem Werkzeug,
    - Sätze des Modells, die Zahlen oder Vergleiche zu den verglichenen Wörtern enthalten, werden entfernt.
    """
    blocks, compared = [], set()
    for name, kwargs, output in steps:
        lines = output.splitlines()
        table = "\n".join(line for line in lines if line.startswith("|"))
        summary = next((line for line in lines if line.startswith("Kurzfassung (berechnet):")), "")
        if table and table not in blocks:
            blocks.append(table + (f"\n\n**{summary.split(':', 1)[0]}:**{summary.split(':', 1)[1]}" if summary else ""))
        if name == "compare_years":
            compared.update(t.casefold() for t in tools.split_terms(str(kwargs.get("terms", ""))))
    if not blocks:
        return text
    if compared:
        # Zahlen, die wörtlich in anderen Werkzeugergebnissen stehen (z. B. Ortsregister), dürfen bleiben.
        allowed = {n for name, _, out in steps if name != "compare_years" for n in NUMBER.findall(out)}

        def doubtful(sentence: str) -> bool:
            if not any(t in sentence.casefold() for t in compared):
                return False
            return bool(COMPARATIVE.search(sentence)) or any(n not in allowed for n in NUMBER.findall(sentence))

        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = [x for x in sentences if not doubtful(x)]
        text = " ".join(kept).strip() or "Hier der Vergleich aus dem Volltext:"
    return text + "\n\n" + "\n\n".join(blocks) + "\n\n_Zahlen und Kurzfassung direkt aus dem Werkzeug (Volltext, automatisch erkannt)._"


def ask(question: str, history: list[tuple[str, str]] | None = None) -> AgentAnswer:
    """Eine Frage stellen; history = frühere (Rolle, Text)-Paare für Rückfragen wie „Und bei Dorpat?“."""
    return asyncio.run(_run(question, history))


if __name__ == "__main__":
    import os
    import sys
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    result = ask(" ".join(sys.argv[1:]) or "Wie oft werden Reval und Tallinn in den Jahrgängen genannt?")
    for name, kwargs, output in result.steps:
        print(f"→ {name}({kwargs})\n  {output[:300]}\n")
    print(result.text)
