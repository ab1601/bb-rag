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
    # streng nach Häufigkeit sortiert – sonst schließt ein Sprachmodell aus der Reihenfolge auf die Häufigkeit
    forms.sort(key=lambda p: (-sum(corpus.place_mentions.get(p["name"], Counter()).values()), p["name"]))
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
    return (f"{len(forms)} Ortsform(en) mit „{name}“, nach Nennungen absteigend sortiert. Formen werden nur bei identischer Schreibweise "
            f"zusammengeführt; Zuordnungen sind automatisch und ungeprüft.\n" + "\n".join(lines) + more)


def split_terms(terms: str) -> list[str]:
    """ "Reval, Tallinn" / "Reval und Tallinn" / "Reval; Tallinn" → ["Reval", "Tallinn"]"""
    parts = re.split(r"\s*(?:,|;|\bund\b|\boder\b|\bvs\.?|\bgegen\b)\s*", terms.strip())
    return list(dict.fromkeys(p.strip(" \"'„“") for p in parts if p.strip(" \"'„“")))


def compare_years(terms: str, path: str | None = None) -> str:
    """Häufigkeit eines oder mehrerer Wörter je Jahrgang als Tabelle: Treffer und Treffer je 100.000 Wörter."""
    words = split_terms(terms or "")
    if not words:
        return "Kein Suchbegriff angegeben."
    corpus = load_corpus(path)
    counts = {}
    for w in words:
        pattern = re.compile(rf"(?<!\w){re.escape(w)}(?!\w)", re.IGNORECASE)
        counts[w] = Counter()
        for d in corpus.documents:
            counts[w][int(d["year"])] += len(pattern.findall(d["text"]))
    header = "| Jahrgang | " + " | ".join(f"{w} (Treffer / je 100.000 Wörter)" for w in words) + " |"
    lines = [header, "|---" * (len(words) + 1) + "|"]
    for y in sorted(corpus.words_per_year):
        n = corpus.words_per_year[y]
        cells = [f"{counts[w][y]} / {counts[w][y] / n * 1e5:.1f}" if n else "0 / 0.0" for w in words]
        lines.append(f"| {y} | " + " | ".join(cells) + " |")
    lines.append("| Summe | " + " | ".join(str(sum(counts[w].values())) for w in words) + " |")
    lines.append("")
    lines.append("Kurzfassung (berechnet): " + summarize_counts(counts, sorted(corpus.words_per_year)))
    return ("Jahrgangsvergleich (exakte Schreibweise im Volltext, auch in Doppelformen wie Reval/Tallinn). "
            "Die Tabelle wird automatisch unter deine Antwort gesetzt. Schreibe selbst keine Zahlen daraus ab, "
            "sondern nur einen kurzen einleitenden Satz:\n" + "\n".join(lines))


def summarize_counts(counts: dict[str, Counter], years: list[int]) -> str:
    """Deterministische Kurzfassung eines Jahrgangsvergleichs – Aussagen, die ein Modell oft falsch formuliert."""
    words = list(counts)
    parts = ["Insgesamt " + ", ".join(f"„{w}“ {sum(counts[w].values())}" for w in words) + " Treffer."]
    for w in words:
        present = [y for y in years if counts[w][y]]
        if not present:
            parts.append(f"„{w}“ kommt nicht vor.")
        elif len(present) == len(years):
            parts.append(f"„{w}“ erscheint in allen {len(years)} Jahrgängen.")
        else:
            parts.append(f"„{w}“ erscheint nur {', '.join(map(str, present))}.")
    if len(words) == 2:
        a, b = words
        more_b = [y for y in years if counts[b][y] > counts[a][y]]
        equal = [y for y in years if counts[b][y] == counts[a][y] and counts[a][y]]
        if more_b:
            parts.append(f"„{b}“ liegt vor „{a}“ in: {', '.join(map(str, more_b))}.")
        else:
            parts.append(f"„{b}“ liegt in keinem Jahrgang vor „{a}“.")
        if equal:
            parts.append(f"Gleichstand: {', '.join(map(str, equal))}.")
    return " ".join(parts)
