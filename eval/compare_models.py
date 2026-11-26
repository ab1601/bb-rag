"""Vergleicht Sprachmodelle im Agent-Modus anhand fester Testfragen (lokal, braucht Ollama und den Index).

Aufruf:  BB_LLM_MODEL=qwen2.5:7b python -m eval.compare_models
Geprüft wird: richtiges Werkzeug, erwartete Argumente/Belege, Antwortzeit und – wichtig – der ROHE Text des
Modells vor dem Faktenfilter, um zu sehen, wie oft das Modell selbst Zahlen falsch wiedergibt.
Die Ausgabe enthält Ausschnitte aus den Texten und wird deshalb nicht ins Repository übernommen.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from bb_rag import config
from bb_rag.agent import stream

CASES = [
    {"id": "vergleich", "q": "Wie oft werden Reval und Tallinn in den einzelnen Jahrgängen genannt?",
     "tool": "compare_years", "arg": ["reval", "tallinn"],
     "wrong": [r"tallinn[^.]*nur[^.]*1988 und 1991", r"tallinn[^.]*häufiger[^.]*als[^.]*reval(?![^.]*2004)"]},
    {"id": "formen", "q": "Welche Schreibformen gibt es für Tartu, und wie oft wird Tartu im Vergleich zu Dorpat genannt?",
     "tool": "place_register", "arg": ["tartu"],
     "wrong": [r"tartu[^.]*häufiger[^.]*als[^.]*dorpat", r"häufigste form ist[^.]*„?\"?tartu\"?“? selbst"]},
    {"id": "register", "q": "Welche Koordinate hat Riga im Ortsregister, und ist sie geprüft?",
     "tool": "place_register", "arg": ["riga"], "expect": [r"56[.,]9"], "wrong": [r"\bgeprüft\b(?![^.]*ungeprüft)(?<!un)"]},
    {"id": "inhalt", "q": "Was wird 1949 über die Jugendtagung auf dem Ludwigstein berichtet?",
     "tool": "semantic_search", "arg": ["ludwigstein"], "expect": [r"bb_11_49"]},
    {"id": "ungueltig", "q": "Was steht im Jahrgang 1975 über Riga?",
     "tool": None, "expect": [r"1975[^.]*(nicht|kein)|(nicht|kein)[^.]*1975|vorhanden sind|nur die Jahrgänge"]},
]
FOLLOW = {"id": "rueckfrage", "first": "Wie oft wird Reval in den einzelnen Jahrgängen genannt?",
          "q": "Und wie ist es bei Dorpat?", "tool": None, "arg": ["dorpat"]}


async def run(question, history=None):
    raw, steps, final = "", [], None
    started = time.time()
    async for ev in stream(question, history):
        if ev["type"] == "delta":
            raw += ev["text"]
        elif ev["type"] == "tool_result":
            steps.append((ev["name"], ev["kwargs"]))
        elif ev["type"] == "done":
            final = ev["answer"]
    return raw.strip(), steps, final, time.time() - started


def check(case, raw, steps):
    notes, ok = [], True
    names = [n for n, _ in steps]
    args = json.dumps([a for _, a in steps], ensure_ascii=False).casefold()
    if case.get("tool") and case["tool"] not in names:
        ok = False; notes.append(f"Werkzeug {case['tool']} fehlt")
    for a in case.get("arg", []):
        if a not in args:
            ok = False; notes.append(f"Argument „{a}“ fehlt")
    for pat in case.get("expect", []):
        if not re.search(pat, raw, re.I):
            ok = False; notes.append(f"erwartet /{pat}/")
    for pat in case.get("wrong", []):
        if re.search(pat, raw, re.I):
            ok = False; notes.append(f"FALSCHE AUSSAGE /{pat}/")
    return ok, notes


async def main():
    results = []
    for case in CASES:
        raw, steps, _, secs = await run(case["q"])
        results.append((case["id"], *check(case, raw, steps), secs, steps, raw))
    first_raw, _, first, _ = await run(FOLLOW["first"])
    raw, steps, _, secs = await run(FOLLOW["q"], [("user", FOLLOW["first"]), ("assistant", first.text)])
    results.append((FOLLOW["id"], *check(FOLLOW, raw, steps), secs, steps, raw))

    print(f"\n=== Modell: {config.LLM_MODEL} ===")
    for cid, ok, notes, secs, steps, raw in results:
        print(f"[{'OK ' if ok else 'XX '}] {cid:11s} {secs:5.1f}s  {[(n, a) for n, a in steps]}")
        for n in notes:
            print(f"       - {n}")
        print("       roh:", raw.replace("\n", " ")[:260])
    passed = sum(r[1] for r in results)
    print(f"Ergebnis: {passed}/{len(results)} bestanden, Ø {sum(r[3] for r in results) / len(results):.1f} s")


if __name__ == "__main__":
    asyncio.run(main())
