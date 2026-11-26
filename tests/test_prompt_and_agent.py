import os

import pytest

from bb_rag import config
from bb_rag.search import Hit


def _hit(n, year=1949):
    return Hit(f"t_01_49#{n:03d}", 0.9, year, "01", "Testblatt · 01/1949", f"Text {n}", [], [])


def test_prompt_numbers_sources():
    from bb_rag.rag import SYSTEM, build_prompt
    prompt = build_prompt("Wer sprach?", [_hit(0), _hit(1)])
    assert "[1] Testblatt · 01/1949, Jahrgang 1949\nText 0" in prompt
    assert "[2] Testblatt" in prompt and "[3]" not in prompt
    assert prompt.rstrip().endswith("Antwort (mit Belegen in eckigen Klammern):")
    assert "nur auf Grundlage" in SYSTEM


def test_hit_citation():
    assert _hit(3).cite() == "Testblatt · 01/1949 [t_01_49#003]"


def test_agent_tools_are_described():
    from bb_rag.agent import SYSTEM, TOOLS
    names = {t.metadata.name for t in TOOLS}
    assert names == {"semantic_search", "fulltext_search", "place_register", "compare_years"}
    for t in TOOLS:
        assert "Args:" in t.metadata.description
        assert t.metadata.name in SYSTEM


@pytest.mark.integration
def test_agent_end_to_end():
    """Braucht Ollama mit llama3.1:8b und den echten Index – läuft nur lokal: pytest -m integration."""
    if not (config.STORAGE / "chroma").exists() or os.getenv("CI"):
        pytest.skip("kein lokaler Index / CI")
    from bb_rag.agent import ask
    result = ask("Wie oft wird Tallinn in den einzelnen Jahrgängen genannt?")
    assert result.steps, "der Agent sollte mindestens ein Werkzeug aufrufen"
    assert "1988" in result.text


def test_history_strips_tool_log_and_limits_turns():
    from bb_rag.agent import MAX_TURNS, history_messages
    hist = [("user", "Wie oft Reval?"), ("assistant", "118 Mal.<details><summary>x</summary>lang</details>")] * 5
    msgs = history_messages(hist)
    assert len(msgs) == MAX_TURNS
    assert all("<details>" not in m.content for m in msgs)
    assert msgs[-1].content == "118 Mal."
    assert history_messages(None) == []


def test_tables_from_tools_are_attached_once():
    from bb_rag.agent import attach_tables
    table_out = "Jahrgangsvergleich:\n| Jahrgang | Reval |\n|---|---|\n| 1949 | 19 / 38.8 |"
    steps = [("compare_years", {"terms": "Reval"}, table_out), ("compare_years", {"terms": "Reval"}, table_out)]
    text = attach_tables("Die Tabelle zeigt die Treffer.", steps)
    assert text.count("| 1949 | 19 / 38.8 |") == 1
    assert attach_tables("Antwort ohne Werkzeugtabelle.", [("place_register", {}, "- Reval: 118")]) == "Antwort ohne Werkzeugtabelle."


def test_model_comparisons_are_replaced_by_computed_summary():
    from bb_rag import tools
    from bb_rag.agent import attach_tables
    out = tools.compare_years("Reval, Tallinn", path="tests/fixtures/mini_corpus.json")
    steps = [("place_register", {"name": "Reval"}, "2 Ortsform(en) mit „Reval“"), ("compare_years", {"terms": "Reval, Tallinn"}, out)]
    model = "Es gibt 2 Formen von Reval. Tallinn wird häufiger genannt als Reval. Reval erscheint 7 Mal. Die Daten sind ungeprüft."
    text = attach_tables(model, steps)
    assert "Es gibt 2 Formen von Reval." in text          # Zahl steht im Ortsregister → bleibt
    assert "häufiger" not in text and "7 Mal" not in text  # Vergleich bzw. erfundene Zahl → entfernt
    assert "Die Daten sind ungeprüft." in text
    assert "Kurzfassung (berechnet):" in text and "„Tallinn“ liegt vor „Reval“ in: 1988." in text


def test_summarize_counts():
    from collections import Counter
    from bb_rag.tools import summarize_counts
    s = summarize_counts({"A": Counter({1: 2, 2: 1}), "B": Counter({2: 1, 3: 5})}, [1, 2, 3])
    assert s == ("Insgesamt „A“ 3, „B“ 6 Treffer. „A“ erscheint nur 1, 2. „B“ erscheint nur 2, 3. "
                 "„B“ liegt vor „A“ in: 3. Gleichstand: 2.")
