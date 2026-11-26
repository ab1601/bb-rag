"""Zerlegt die Ausgaben in Abschnitte mit Metadaten (Ausgabe, Jahr, Orte, Personen im Abschnitt).

Bewusst ohne LlamaIndex-Abhängigkeit, damit die Logik einfach testbar ist.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    year: int
    issue: str
    title: str
    start: int          # Zeichenposition im Ausgabentext
    end: int
    text: str
    places: list[str] = field(default_factory=list)
    persons: list[str] = field(default_factory=list)

    def metadata(self) -> dict:
        # Chroma speichert nur einfache Werte: Listen als Text mit Trennzeichen.
        return {
            "doc_id": self.doc_id, "year": self.year, "issue": self.issue, "title": self.title,
            "start": self.start, "end": self.end,
            "places": " | ".join(self.places), "persons": " | ".join(self.persons),
        }


def load_documents(path: Path | None = None) -> list[dict]:
    data = json.loads(Path(path or config.CORPUS).read_text(encoding="utf-8"))
    return [d for d in data["documents"] if not d.get("fictional")]


def _paragraphs(text: str) -> list[tuple[int, int]]:
    """Absatzgrenzen (Start, Ende) im Originaltext; Leerzeilen trennen Absätze."""
    spans, pos = [], 0
    for match in re.finditer(r"\n\s*\n", text):
        if match.start() > pos:
            spans.append((pos, match.start()))
        pos = match.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def _split_long(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Überlange Absätze an Satzgrenzen teilen."""
    if end - start <= limit:
        return [(start, end)]
    out, cur = [], start
    for match in re.finditer(r"(?<=[.!?…])\s+", text[start:end]):
        cut = start + match.end()
        if cut - cur >= limit * 0.6:
            out.append((cur, cut))
            cur = cut
    if cur < end:
        out.append((cur, end))
    return out


def chunk_document(doc: dict, target: int = config.CHUNK_TARGET, limit: int = config.CHUNK_MAX) -> list[Chunk]:
    text = doc["text"]
    pieces = [s for a, b in _paragraphs(text) for s in _split_long(text, a, b, limit)]
    groups, cur = [], []
    for a, b in pieces:
        if cur and (b - cur[0][0]) > target:
            groups.append((cur[0][0], cur[-1][1]))
            cur = []
        cur.append((a, b))
    if cur:
        groups.append((cur[0][0], cur[-1][1]))

    chunks = []
    for n, (a, b) in enumerate(groups):
        ents = [e for e in doc.get("entities", []) if e["start"] >= a and e["end"] <= b]
        uniq = lambda kind: list(dict.fromkeys(e["label"].strip() for e in ents if e["type"] == kind and e["label"].strip()))
        chunks.append(Chunk(
            chunk_id=f"{doc['id']}#{n:03d}", doc_id=doc["id"], year=int(doc["year"]), issue=str(doc["issue"]),
            title=doc.get("title", doc["id"]), start=a, end=b, text=text[a:b].strip(),
            places=uniq("place"), persons=uniq("person"),
        ))
    return chunks


def all_chunks(path: Path | None = None) -> list[Chunk]:
    return [c for d in load_documents(path) for c in chunk_document(d)]
