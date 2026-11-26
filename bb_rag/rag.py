"""RAG: Frage → passende Abschnitte (Retrieval) → Antwort des lokalen Sprachmodells mit Quellenangaben.

Bewusst ohne „Magie“: Die gefundenen Abschnitte werden nummeriert in den Prompt gelegt, das Modell muss
mit [1], [2] … zitieren und darf nur aus diesen Abschnitten antworten. So bleibt nachprüfbar, woher eine
Aussage stammt.
"""
from __future__ import annotations

from dataclasses import dataclass

from llama_index.llms.ollama import Ollama

from . import config
from .search import Hit, search

SYSTEM = (
    "Du beantwortest Fragen zur Zeitschrift „Baltische Briefe“ (deutschbaltische Diaspora, 1949–2004). "
    "Antworte auf Deutsch, knapp und sachlich, und nur auf Grundlage der nummerierten Textabschnitte. "
    "Belege jede Aussage mit der Nummer des Abschnitts in eckigen Klammern, z. B. [2]. "
    "Wenn die Abschnitte die Frage nicht beantworten, sage das ausdrücklich. "
    "Die Texte stammen aus OCR und können Lesefehler enthalten."
)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Hit]
    model: str


def _llm() -> Ollama:
    return Ollama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL, request_timeout=300.0,
                  temperature=0.0, context_window=8192)


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = "\n\n".join(f"[{i}] {h.title}, Jahrgang {h.year}\n{h.text}" for i, h in enumerate(hits, start=1))
    return f"Textabschnitte:\n\n{blocks}\n\nFrage: {question}\n\nAntwort (mit Belegen in eckigen Klammern):"


def answer(question: str, year: int | None = None, place: str | None = None, top_k: int = config.TOP_K) -> Answer:
    hits = search(question, year=year, place=place, top_k=top_k)
    if not hits:
        return Answer(question, "Zu dieser Frage wurden keine passenden Abschnitte gefunden.", [], config.LLM_MODEL)
    from llama_index.core.llms import ChatMessage
    response = _llm().chat([ChatMessage(role="system", content=SYSTEM),
                            ChatMessage(role="user", content=build_prompt(question, hits))])
    return Answer(question, response.message.content.strip(), hits, config.LLM_MODEL)


if __name__ == "__main__":
    import sys
    a = answer(" ".join(sys.argv[1:]) or "Wie wird 1991 über Reisen ins Baltikum berichtet?")
    print(a.text, "\n")
    for i, h in enumerate(a.sources, start=1):
        print(f"[{i}] {h.cite()}  (Ähnlichkeit {h.score})")
