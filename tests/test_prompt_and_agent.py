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
