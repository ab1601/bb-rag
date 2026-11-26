# bb-rag – Fragen an die „Baltischen Briefe“

[![CI](https://github.com/ab1601/bb-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ab1601/bb-rag/actions/workflows/ci.yml)

Lokales **RAG- und Agentensystem** über die Zeitschrift *Baltische Briefe* (deutschbaltische Diaspora):
48 Ausgaben aus den Jahrgängen 1949, 1950, 1988, 1991 und 2004, rund 290.000 Wörter, OCR-Text mit
automatisch erkannten Orts- und Personennamen.

Alles läuft **lokal**: Embeddings, Vektordatenbank und Sprachmodell. Die Texte verlassen den Rechner nicht.

![Chat-Oberfläche](docs/screenshot.png)

## Was es kann

| Modus | Ablauf | Gut für |
|---|---|---|
| **RAG** | Frage → semantische Suche (Chroma) → Llama 3.1 antwortet nur aus den gefundenen Abschnitten, mit Belegen [1], [2] … | inhaltliche Fragen |
| **Agent** | Llama 3.1 wählt selbst Werkzeuge, ruft sie auf und antwortet mit Belegen; jeder Aufruf wird angezeigt; Rückfragen („Und bei Dorpat?“) beziehen das Gespräch ein | Zählen, Vergleichen, Register, kombinierte Fragen |

Werkzeuge des Agenten (`bb_rag/agent.py`, `bb_rag/tools.py`):

- `semantic_search(query, year, place)` – Vektorsuche, optional nach Jahrgang oder Ort gefiltert
- `fulltext_search(term, year)` – exakte Wortsuche mit Treffern pro Jahrgang und Textstellen
- `place_register(name)` – alle Schreibformen eines Ortes mit Nennungen, Prüfstatus, Koordinate, Wikidata
- `compare_years(terms)` – Häufigkeit eines oder mehrerer Wörter je Jahrgang als Tabelle, absolut und je 100.000 Wörter

Beispiel: *„Wie oft werden Reval und Tallinn in den einzelnen Jahrgängen genannt?“* → der Agent ruft
`compare_years("Reval, Tallinn")` auf und antwortet mit einer Tabelle exakter Zahlen statt mit Schätzungen.

### Zahlen kommen aus dem Code, nicht aus dem Sprachmodell

Beim Testen zeigte sich, dass ein 8B-Modell korrekte Werkzeugergebnisse falsch zusammenfasst – etwa „Tallinn
wird nur 1988 und 1991 erwähnt“, obwohl die Tabelle auch 2004 enthielt – oder nach einem Werkzeugfehler Zahlen
erfindet. Deshalb:

- Werkzeuge liefern Zahlen als fertige Tabellen; der Code hängt sie unverändert an die Antwort (`attach_tables`).
- Fehlerhafte Argumente des Modells (z. B. Jahrgang `"None"`) werden abgefangen und als klare Meldung zurückgegeben.
- Der Prompt verbietet Zahlen, die in keinem Werkzeugergebnis stehen; alle Aufrufe sind in der Oberfläche sichtbar.

## Architektur

```
research-corpus.json ──► chunking.py ──► index.py ──► Chroma (storage/chroma)
 (Text + NER-Entitäten)   ~2.100 Abschnitte   Qwen3-Embedding-0.6B          │
                          mit Jahr, Ausgabe,  (Hugging Face, lokal)          ▼
                          Orten, Personen                     search.py ──► rag.py ──► Llama 3.1 8B (Ollama)
                                                                  │                         ▲
                                         tools.py ────────────► agent.py (LlamaIndex FunctionAgent)
                                                                  │
                                                               app.py (Gradio)
```

| Baustein | Technik |
|---|---|
| Framework | LlamaIndex (VectorStoreIndex, FunctionAgent, FunctionTool) |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` über Hugging Face / sentence-transformers (MPS oder CPU) |
| Vektordatenbank | Chroma, persistent, Kosinus-Ähnlichkeit |
| Sprachmodell | Llama 3.1 8B (Q4) über Ollama, Temperatur 0 |
| Oberfläche | Gradio |
| Tests / CI | pytest, GitHub Actions (Tests, Docker-Build, Prüfung der Kubernetes-Manifeste) |
| Betrieb | Docker, docker compose, Kubernetes (Deployments, Services, PVCs, Jobs) |

## Start (macOS, lokal)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
ollama pull llama3.1:8b                    # Ollama-App muss laufen
.venv/bin/python -m bb_rag.index           # Index bauen (~3 Min. auf Apple Silicon)
.venv/bin/python app.py                    # http://127.0.0.1:7860
.venv/bin/python -m bb_rag.agent "Welche Schreibformen gibt es für Tartu?"   # Agent im Terminal
```

Der Pfad zum Korpus wird über `BB_CORPUS` gesetzt (Standard: `../baltische-briefe-portal/data/research-corpus.json`).

## Docker

```bash
docker compose up -d
docker compose exec ollama ollama pull llama3.1:8b
docker compose run --rm app python -m bb_rag.index
# → http://127.0.0.1:7860
```

Korpus und Index werden als Volumes eingebunden, nicht ins Image kopiert.

**macOS:** Docker hat keinen Zugriff auf die Apple-GPU. Schneller ist es, die Ollama-App und das bereits
geladene Embedding-Modell des Mac zu nutzen – dafür gibt es eine eigene Compose-Datei (getestet mit
Docker Desktop 4.100 auf Apple Silicon):

```bash
docker compose -f docker-compose.yml -f docker-compose.mac.yml up -d --build app
# → http://127.0.0.1:7861
```

## Kubernetes

Manifeste in `k8s/` (Namespace, PersistentVolumeClaims, Deployments und Services für App und Ollama,
ConfigMap, Readiness- und Liveness-Probes, Jobs für Modell-Download und Indexaufbau). Getestet für ein
lokales Cluster (Docker Desktop oder kind); den Korpuspfad in `k8s/app.yaml` und `k8s/jobs.yaml` anpassen.

```bash
docker build -t bb-rag:latest .
kubectl apply -k k8s/                      # App, Ollama, Speicher
kubectl apply -f k8s/jobs.yaml             # Llama 3.1 laden, Index bauen
kubectl -n bb-rag port-forward svc/bb-rag 7860:7860
```

Die CI prüft die Manifeste bei jedem Push mit `kubeconform`.

## Tests

```bash
.venv/bin/python -m pytest                 # 19 Tests, erfundenes Mini-Korpus, ohne Sprachmodell
.venv/bin/python -m pytest -m integration  # Ende-zu-Ende mit Ollama und echtem Index
```

Die Unit-Tests laufen auf `tests/fixtures/mini_corpus.json` – einem **erfundenen** Testkorpus. So laufen sie
auch in GitHub Actions, ohne dass Texte der Zeitschrift ins Repository gelangen.

## Daten und Rechte

Die Texte der *Baltischen Briefe* sind urheberrechtlich geschützt; die Rechte für eine Veröffentlichung sind
nicht geklärt. Deshalb enthält dieses Repository **keine Texte, keine Scans und keinen Index** (`storage/` und
`data/` stehen in `.gitignore`). Das System ist für die Forschung im nicht öffentlichen Rahmen gedacht.

## Grenzen

- OCR-Fehler und Silbentrennungen (*Liv- land*) stören Suche und Erkennung.
- Ortsnamen sind automatisch und **ungeprüft** zugeordnet; Schreibformen werden nur bei identischer
  Schreibweise zusammengeführt (*Dorpat*, *Tartu* und *Dorpat/Tartu* sind drei Einträge).
- Ein 8B-Modell wählt Werkzeuge meist sinnvoll, aber nicht immer; deshalb werden alle Aufrufe angezeigt.
- Volltextzählung (exakte Zeichenkette) und Registerzählung (NER-Nennungen) können voneinander abweichen.
