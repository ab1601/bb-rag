from bb_rag import tools


def test_fulltext_counts_per_year(corpus_path):
    out = tools.fulltext_search("Tallinn", path=corpus_path)
    # "Tallinn" zweimal allein, einmal in "Reval/Tallinn" – Wortgrenze am Schrägstrich zählt mit
    assert "3 Treffer (1988: 3)" in out
    assert "t_02_88" in out


def test_fulltext_respects_year_and_case(corpus_path):
    assert "kommt im Korpus im Jahrgang 1949 nicht vor" in tools.fulltext_search("tallinn", year=1949, path=corpus_path)
    assert "1 Treffer" in tools.fulltext_search("hamburg", path=corpus_path)


def test_fulltext_no_partial_words(corpus_path):
    # "Muster" ist nur Teil von "Musterstadt"
    assert "nicht vor" in tools.fulltext_search("Muster", path=corpus_path)


def test_place_register_status_and_coordinates(corpus_path):
    out = tools.place_register("Reval", path=corpus_path)
    assert out.startswith("2 Ortsform(en)") and "absteigend sortiert" in out
    first = out.splitlines()[1]
    assert first.startswith("- Reval: 1 Nennungen (1949: 1)")
    assert "59.437, 24.745" in first and "Q1770" in first
    assert "Reval/Tallinn" in out and "ohne Koordinate" in out


def test_place_register_ambiguous_and_unknown(corpus_path):
    assert "mehrdeutig" in tools.place_register("Musterstadt", path=corpus_path)
    assert "Keine Ortsform" in tools.place_register("Atlantis", path=corpus_path)


def test_compare_years_table(corpus_path):
    out = tools.compare_years("Reval, Tallinn", path=corpus_path)
    corpus = tools.load_corpus(corpus_path)
    n88 = corpus.words_per_year[1988]
    assert "| Jahrgang | Reval (Treffer / je 100.000 Wörter) | Tallinn (Treffer / je 100.000 Wörter) |" in out
    assert "keine Zahlen daraus ab" in out
    assert f"| 1988 | 1 / {1 / n88 * 1e5:.1f} | 3 / {3 / n88 * 1e5:.1f} |" in out
    assert "| Summe | 2 | 3 |" in out


def test_split_terms():
    assert tools.split_terms("Reval und Tallinn") == ["Reval", "Tallinn"]
    assert tools.split_terms("„Dorpat“; Tartu, Dorpat") == ["Dorpat", "Tartu"]


def test_empty_input():
    assert tools.fulltext_search("  ") == "Kein Suchbegriff angegeben."
    assert tools.place_register("") == "Kein Ortsname angegeben."


def test_year_normalisation(corpus_path):
    assert tools.normalize_year("None") is None and tools.normalize_year("alle") is None
    assert tools.normalize_year("1988") == 1988
    assert "nicht im Korpus" in tools.fulltext_search("Reval", year="1975", path=corpus_path)
    assert "2 Treffer (1949: 1, 1988: 1)" in tools.fulltext_search("Reval", year="None", path=corpus_path)
