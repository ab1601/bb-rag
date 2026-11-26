"""Agent mit Werkzeugen (LlamaIndex FunctionAgent + Llama 3.1 über Ollama, lokal).

Das Sprachmodell entscheidet selbst, welche Werkzeuge es aufruft:
  semantische Suche (Vektordatenbank) · Volltextsuche · Ortsregister · Jahrgangsvergleich.
Jeder Werkzeugaufruf wird protokolliert, damit nachvollziehbar bleibt, worauf eine Antwort beruht.

Aufruf:  python -m bb_rag.agent "Wie oft wird Tallinn im Vergleich zu Reval genannt?"
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
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
    "Nenne Zahlen nur, wenn sie in einem Werkzeugergebnis stehen. Meldet ein Werkzeug einen Fehler, "
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


def compare_years(term: str) -> str:
    """Vergleicht, wie häufig ein Wort in den fünf Jahrgängen vorkommt (absolut und je 100.000 Wörter).

    Args:
        term: Wort in genauer Schreibweise, z. B. "Tallinn".
    """
    return tools.compare_years(term)


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


async def _run(question: str, history: list[tuple[str, str]] | None = None) -> AgentAnswer:
    past = history_messages(history)
    handler = build_agent().run(user_msg=question, chat_history=past or None)
    steps = []
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            steps.append((event.tool_name, dict(event.tool_kwargs), str(event.tool_output)))
    response = await handler
    return AgentAnswer(question, str(response).strip(), steps)


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
