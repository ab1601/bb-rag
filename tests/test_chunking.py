from bb_rag.chunking import all_chunks, chunk_document, load_documents


def test_documents_load(corpus_path):
    docs = load_documents(corpus_path)
    assert [d["id"] for d in docs] == ["t_01_49", "t_02_88"]


def test_chunks_are_text_spans(corpus_path):
    for doc in load_documents(corpus_path):
        for c in chunk_document(doc, target=150, limit=200):
            assert doc["text"][c.start:c.end].strip() == c.text
            assert c.chunk_id.startswith(doc["id"] + "#")


def test_long_paragraph_is_split(corpus_path):
    doc = load_documents(corpus_path)[0]
    chunks = chunk_document(doc, target=150, limit=200)
    assert len(chunks) > 2
    assert all(len(c.text) <= 260 for c in chunks)


def test_entities_go_to_their_chunk(corpus_path):
    doc = load_documents(corpus_path)[0]
    chunks = chunk_document(doc, target=150, limit=200)
    first = chunks[0]
    assert "Hamburg" in first.places and "Reval" in first.places
    assert all("Anton Beispiel" not in c.persons for c in chunks[:-1])
    assert "Anton Beispiel" in chunks[-1].persons


def test_metadata_is_flat(corpus_path):
    meta = all_chunks(corpus_path)[0].metadata()
    assert all(isinstance(v, (str, int)) for v in meta.values())
    assert meta["year"] == 1949
