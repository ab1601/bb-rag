"""Zentrale Einstellungen. Alles lokal: Daten, Embeddings, Vektordatenbank und Sprachmodell (Ollama)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Die Texte bleiben im Portal-Ordner und werden nicht ins Projekt kopiert (Rechte ungeklärt).
CORPUS = Path(os.getenv("BB_CORPUS", ROOT.parent / "baltische-briefe-portal" / "data" / "research-corpus.json"))
STORAGE = Path(os.getenv("BB_STORAGE", ROOT / "storage"))
COLLECTION = "baltische_briefe"

# Embeddings lokal (sentence-transformers); Qwen3-Embedding ist mehrsprachig und auf Deutsch gut.
EMBED_MODEL = os.getenv("BB_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DEVICE = os.getenv("BB_EMBED_DEVICE", "mps")
# Ablageort der Embedding-Modelle; ohne Angabe nutzt LlamaIndex seinen Standard-Cache.
MODEL_CACHE = os.getenv("BB_MODEL_CACHE") or None
QUERY_INSTRUCTION = ("Instruct: Given a question about the German-language newspaper 'Baltische Briefe' "
                     "(Baltic German diaspora, 1949–2004), retrieve passages that answer it\nQuery: ")

# Sprachmodell über Ollama (lokal auf dem Mac, Standardadresse).
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("BB_LLM_MODEL", "llama3.1:8b")

# Abschnitte: Absätze werden zu Stücken von etwa dieser Länge zusammengefasst (Zeichen).
CHUNK_TARGET = 1200
CHUNK_MAX = 2000
TOP_K = 6
