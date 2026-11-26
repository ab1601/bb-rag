"""Vektordatenbank (Chroma, lokal auf der Festplatte) aufbauen und laden.

Aufruf:  python -m bb_rag.index      (baut den Index neu)
"""
from __future__ import annotations

import time

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from . import config
from .chunking import all_chunks


def embedding_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL, device=config.EMBED_DEVICE,
        query_instruction=config.QUERY_INSTRUCTION, text_instruction="",
        embed_batch_size=16, cache_folder=config.MODEL_CACHE,
    )


def _vector_store(reset: bool = False) -> ChromaVectorStore:
    client = chromadb.PersistentClient(path=str(config.STORAGE / "chroma"))
    if reset:
        try:
            client.delete_collection(config.COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    return ChromaVectorStore(chroma_collection=collection)


def build() -> VectorStoreIndex:
    Settings.embed_model = embedding_model()
    chunks = all_chunks()
    nodes = [TextNode(id_=c.chunk_id, text=c.text, metadata=c.metadata(),
                      # Metadaten nicht mit einbetten – nur der Text bestimmt die Ähnlichkeit.
                      excluded_embed_metadata_keys=list(c.metadata()),
                      excluded_llm_metadata_keys=["start", "end"])
             for c in chunks]
    started = time.time()
    store = _vector_store(reset=True)
    index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store),
                             show_progress=True)
    print(f"{len(nodes)} Abschnitte aus {len({c.doc_id for c in chunks})} Ausgaben eingebettet "
          f"({time.time() - started:.0f} s) → {config.STORAGE / 'chroma'}")
    return index


def load() -> VectorStoreIndex:
    Settings.embed_model = embedding_model()
    return VectorStoreIndex.from_vector_store(_vector_store())


if __name__ == "__main__":
    build()
