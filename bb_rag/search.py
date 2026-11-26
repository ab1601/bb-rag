"""Retrieval: semantische Suche in der Vektordatenbank, optional nach Jahrgang oder Ort gefiltert."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from . import config
from .index import load


@dataclass
class Hit:
    chunk_id: str
    score: float
    year: int
    issue: str
    title: str
    text: str
    places: list[str]
    persons: list[str]

    def cite(self) -> str:
        return f"{self.title} [{self.chunk_id}]"


@lru_cache(maxsize=1)
def _index():
    return load()


def search(query: str, year: int | None = None, place: str | None = None, top_k: int = config.TOP_K) -> list[Hit]:
    filters = []
    if year:
        filters.append(MetadataFilter(key="year", value=int(year), operator=FilterOperator.EQ))
    retriever = _index().as_retriever(
        similarity_top_k=top_k * (4 if place else 1),
        filters=MetadataFilters(filters=filters) if filters else None,
    )
    hits = []
    for node in retriever.retrieve(query):
        m = node.metadata
        places = [p for p in m.get("places", "").split(" | ") if p]
        if place and not any(place.lower() in p.lower() for p in places):
            continue
        hits.append(Hit(node.node_id, round(float(node.score or 0), 3), int(m["year"]), m["issue"], m["title"],
                        node.get_content(), places, [p for p in m.get("persons", "").split(" | ") if p]))
    return hits[:top_k]
