"""Werkzeuge für den Agenten: Volltextsuche, Ortsregister und Jahrgangsvergleich über dem Korpus.

Reine Python-Funktionen ohne Modell – sie arbeiten direkt auf research-corpus.json und sind deshalb
einfach testbar. Die semantische Suche (Vektordatenbank) steckt in search.py und wird im Agenten ergänzt.
Alle Funktionen geben kurzen Text zurück, den ein kleines lokales Sprachmodell gut verarbeiten kann.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import config

WORD = re.compile(r"\w+(?:[-'’]\w+)*")
YEARS = (1949, 1950, 1988, 1991, 2004)
STATUS_DE = {"unverified": "automatisch georeferenziert, ungeprüft",
             "ambiguous": "mehrdeutig – mehrere Kandidaten, nicht kartiert",
             "missing": "ohne Koordinate"}


@dataclass(frozen=True)
class Corpus:
    documents: tuple[dict, ...]
    places: tuple[dict, ...]
    words_per_year: dict[int, int]
    place_mentions: dict[str, Counter]        # Ortsform -> Counter(Jahr -> Nennungen)


@lru_cache(maxsize=4)
def load_corpus(path: str | None = None) -> Corpus:
    data = json.loads(Path(path or config.CORPUS).read_text(encoding="utf-8"))
    docs = tuple(d for d in data["documents"] if not d.get("fictional"))
    words, mentions = Counter(), defaultdict(Counter)
    for d in docs:
        words[int(d["year"])] += len(WORD.findall(d["text"]))
        for e in d.get("entities", []):
            if e["type"] == "place":
                mentions[" ".join(e["label"].split())][int(d["year"])] += 1
    return Corpus(docs, tuple(data.get("places", [])), dict(words), dict(mentions))


def normalize_year(year) -> int | None:
    """Sprachmodelle übergeben Jahrgänge oft als Text ("1949", "None", "alle"). Ungültiges → ValueError."""
    if year is None or str(year).strip().lower() in ("", "none", "null", "alle", "all", "0"):
        return None
    y = int(str(year).strip())
    if y not in YEARS:
        raise ValueError(f"Jahrgang {y} ist nicht im Korpus; vorhanden sind {', '.join(map(str, YEARS))}.")
    return y


def _snippet(text: str, start: int, end: int, width: int = 110) -> str:
    a, b = max(0, start - width), min(len(text), end + width)
    s = " ".join(text[a:b].split())
    return ("…" if a else "") + s + ("…" if b < len(text) else "")


def fulltext_search(term: str, year: int | None = None, max_hits: int = 8, path: str | None = None) -> str:
    """Exakte Wortsuche (Groß-/Kleinschreibung egal) mit Trefferzahl pro Jahrgang und Textstellen."""
    term = term.strip()
    if not term:
        return "Kein Suchbegriff angegeben."
    try:
        year = normalize_year(year)
    except ValueError as err:
        return str(err)
    corpus = load_corpus(path)
    pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
    per_year, hits = Counter(), []
    for d in corpus.documents:
        if year and int(d["year"]) != int(year):
            continue
        for m in pattern.finditer(d["text"]):
            per_year[int(d["year"])] += 1
            if len(hits) < max_hits:
                hits.append(f"- {d['id']} ({d['year']}): {_snippet(d['text'], m.start(), m.end())}")
    if not per_year:
        return f"„{term}“ kommt im Korpus{f' im Jahrgang {year}' if year else ''} nicht vor (exakte Schreibweise)."
    counts = ", ".join(f"{y}: {n}" for y, n in sorted(per_year.items()))
    return (f"„{term}“: {sum(per_year.values())} Treffer ({counts}). Erste Textstellen:\n" + "\n".join(hits))


def place_register(name: str, max_forms: int = 8, path: str | None = None) -> str:
    """Ortsregister: Schreibformen, die den Namen enthalten, mit Prüfstatus, Koordinate und Nennungen."""
    name = name.strip()
    if not name:
        return "Kein Ortsname angegeben."
    corpus = load_corpus(path)
    needle = name.casefold()
    forms = [p for p in corpus.places if needle in p["name"].casefold()]
    forms.sort(key=lambda p: (p["name"].casefold() != needle, -sum(corpus.place_mentions.get(p["name"], Counter()).values())))
    if not forms:
        return f"Keine Ortsform mit „{name}“ im Register."
    lines = []
    for p in forms[:max_forms]:
        c = corpus.place_mentions.get(p["name"], Counter())
        cand = p.get("candidates") or []
        wd = cand[0].get("wikidata") if cand else None
        desc = cand[0].get("description") if cand else None
        coord = f"{p['lat']:.3f}, {p['lon']:.3f}" if p.get("lat") is not None else "keine"
        years = ", ".join(f"{y}: {n}" for y, n in sorted(c.items())) or "keine"
        lines.append(f"- {p['name']}: {sum(c.values())} Nennungen ({years}); Status: "
                     f"{STATUS_DE.get(p.get('status'), p.get('status'))}; Koordinate: {coord}"
                     + (f"; Wikidata {wd}" if wd else "") + (f" ({desc})" if desc else ""))
    more = f"\n… und {len(forms) - max_forms} weitere Formen." if len(forms) > max_forms else ""
    return (f"{len(forms)} Ortsform(en) mit „{name}“. Formen werden nur bei identischer Schreibweise "
            f"zusammengeführt; Zuordnungen sind automatisch und ungeprüft.\n" + "\n".join(lines) + more)


def compare_years(term: str, path: str | None = None) -> str:
    """Häufigkeit eines Wortes je Jahrgang, absolut und je 100.000 Wörter (Jahrgänge sind unterschiedlich groß)."""
    term = term.strip()
    if not term:
        return "Kein Suchbegriff angegeben."
    corpus = load_corpus(path)
    pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
    counts = Counter()
    for d in corpus.documents:
        counts[int(d["year"])] += len(pattern.findall(d["text"]))
    rows = []
    for y in sorted(corpus.words_per_year):
        rate = counts[y] / corpus.words_per_year[y] * 1e5 if corpus.words_per_year[y] else 0
        rows.append(f"- {y}: {counts[y]} Treffer, {rate:.1f} je 100.000 Wörter")
    return f"Jahrgangsvergleich für „{term}“ (exakte Schreibweise):\n" + "\n".join(rows)
