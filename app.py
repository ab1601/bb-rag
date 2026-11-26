"""Chat-Oberfläche (Gradio) für die Baltischen Briefe. Läuft lokal.

Zwei Modi:
  RAG    – Frage → semantische Suche → Antwort mit nummerierten Quellen
  Agent  – das Sprachmodell wählt selbst Werkzeuge (semantische Suche, Volltext, Ortsregister, Jahrgangsvergleich)

Start:  .venv/bin/python app.py   →  http://127.0.0.1:7860
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr

from bb_rag.agent import ask
from bb_rag.rag import answer

YEARS = ["alle", "1949", "1950", "1988", "1991", "2004"]
MODES = ["Agent (mit Werkzeugen)", "RAG (semantische Suche)"]


def _rag(message: str, year: str, place: str) -> str:
    result = answer(message, year=None if year == "alle" else int(year), place=place.strip() or None)
    sources = "\n".join(
        f"**[{i}]** {h.title} · `{h.chunk_id}` · Ähnlichkeit {h.score}"
        + (f" · Orte: {', '.join(h.places[:5])}" if h.places else "")
        + f"\n> {h.text[:300].strip()}…"
        for i, h in enumerate(result.sources, start=1)
    )
    return f"{result.text}\n\n---\n**Quellen** (Modell: {result.model}, lokal)\n\n{sources}"


def _agent(message: str, history: list, year: str, place: str) -> str:
    hints = []
    if year != "alle":
        hints.append(f"Jahrgang {year}")
    if place.strip():
        hints.append(f"Ort {place.strip()}")
    question = message + (f" (Einschränkung: {', '.join(hints)})" if hints else "")
    past = [(m["role"], m["content"]) for m in history if isinstance(m, dict) and isinstance(m.get("content"), str)]
    result = ask(question, past)
    steps = "\n\n".join(
        f"**{i}. `{name}`** `{kwargs}`\n```\n{output[:600].strip()}\n```"
        for i, (name, kwargs, output) in enumerate(result.steps, start=1)
    ) or "_Keine Werkzeuge aufgerufen._"
    return (f"{result.text}\n\n<details><summary>Werkzeugaufrufe ({len(result.steps)})</summary>\n\n"
            f"{steps}\n\n</details>")


def respond(message: str, history: list, mode: str, year: str, place: str):
    return _agent(message, history, year, place) if mode.startswith("Agent") else _rag(message, year, place)


with gr.Blocks(title="Baltische Briefe – Fragen an das Archiv") as demo:
    gr.Markdown("## Baltische Briefe – Fragen an das Archiv\n"
                "Lokales RAG- und Agentensystem über 48 Ausgaben (1949, 1950, 1988, 1991, 2004). "
                "Antworten mit Belegen; die Texte stammen aus OCR, Orte sind automatisch und ungeprüft zugeordnet. "
                "Alle Daten und Modelle bleiben auf diesem Rechner.")
    with gr.Row():
        mode = gr.Radio(MODES, value=MODES[0], label="Modus")
        year = gr.Dropdown(YEARS, value="alle", label="Jahrgang")
        place = gr.Textbox(label="Ort (optional, z. B. Riga)")
    gr.ChatInterface(respond, additional_inputs=[mode, year, place], chatbot=gr.Chatbot(height=720, label="Chat"),
                     examples=[["Wie oft werden Reval und Tallinn in den einzelnen Jahrgängen genannt?", MODES[0], "alle", ""],
                               ["Welche Schreibformen gibt es für Tartu und sind sie georeferenziert?", MODES[0], "alle", ""],
                               ["Was wird über die Jugendtagung auf dem Ludwigstein berichtet?", MODES[0], "1949", ""],
                               ["Wie wurde den baltischen Flüchtlingen nach dem Krieg geholfen?", MODES[1], "1949", ""]])

if __name__ == "__main__":
    demo.launch(server_name=os.getenv("HOST", "127.0.0.1"), server_port=int(os.getenv("PORT", "7860")))
