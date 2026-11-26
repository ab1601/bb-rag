from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "mini_corpus.json"


@pytest.fixture
def corpus_path() -> str:
    """Erfundenes Mini-Korpus – die Tests brauchen keine Texte der Baltischen Briefe."""
    return str(FIXTURE)
