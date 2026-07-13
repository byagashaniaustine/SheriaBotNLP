# Sheria-Bot NLP API (FastAPI)

Wraps the Sheria-Bot NLP pipeline as a REST API. Every stage of the pipeline
becomes an HTTP endpoint; every graph is served as a static PNG.

## Layout

```
api/
├── main.py             ← FastAPI app + routes
├── pipeline.py         ← NLP stages (pure Python — no FastAPI dependency)
├── graphs_gen.py       ← PNG generation for each stage
├── schemas.py          ← Pydantic request/response models
├── config.py           ← paths + hyperparameters
├── requirements.txt
├── run.sh              ← one-command launcher
└── artifacts/          ← trained model saved here (created at runtime)
```

Data lives one level up in `../data/04_Intents.csv`. Graphs are written to
`../graphs/*.png` and served at `/graphs/<filename>.png`.

## Install & run

```bash
cd "sheriabot NLP/api"
pip install -r requirements.txt
uvicorn main:app --reload
```

Or one command:

```bash
bash run.sh
```

Then open Swagger UI at **http://127.0.0.1:8000/docs** — you can hit every
endpoint from the browser without writing any HTTP client code.

## Endpoint map

| Method | Path | Purpose |
|---|---|---|
| GET  | `/` | Root — lists available endpoints |
| GET  | `/health` | Health check + dataset status |
| GET  | `/stats` | Stage 4 — EDA numbers (class balance, languages, avg length) |
| GET  | `/nlp/pos` | Stage 11 — POS tag distribution |
| GET  | `/nlp/ner` | Stage 12 — NER entity counts + examples |
| GET  | `/nlp/topwords` | Stage 13 — Top-N word frequencies |
| POST | `/pipeline/features` | Stage 14 — Build BoW + TF-IDF (must run before `/classify`) |
| GET  | `/nlp/tfidf` | Stage 14 — Top 20 TF-IDF terms |
| POST | `/pipeline/word2vec` | Stage 15 — Train Word2Vec |
| GET  | `/nlp/embedding/{word}` | Stage 15 — Vector + similar words for one token |
| GET  | `/nlp/tokenise?text=...` | Stage 16 — NLTK vs multilingual BERT tokens |
| POST | `/pipeline/train` | Train four classifiers, save the best |
| POST | `/classify` | Predict the intent of a legal question |
| POST | `/graphs/generate/{stage}` | Render graphs for a stage (stage04/11/12/13/14/15/16/all) |
| GET  | `/graphs/list` | List every graph PNG currently on disk |
| GET  | `/graphs/{name}.png` | Serve a graph as an image |

## End-to-end walkthrough

Right after starting the API the first time, run this sequence to bring
everything online:

```bash
# 1. Build the feature matrix (needed for /classify and stage 14 graphs)
curl -X POST http://127.0.0.1:8000/pipeline/features

# 2. Train the four classifiers, keep the best one
curl -X POST http://127.0.0.1:8000/pipeline/train

# 3. Train Word2Vec (needed for stage 15 graph + /nlp/embedding)
curl -X POST "http://127.0.0.1:8000/pipeline/word2vec?vector_size=100&epochs=15"

# 4. Generate all graphs at once
curl -X POST http://127.0.0.1:8000/graphs/generate/all

# 5. Classify something
curl -X POST http://127.0.0.1:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Nimefutwa kazi bila notisi, nifanye nini?"}'
```

After step 4 the graphs are all sitting in `../graphs/*.png` and are
browsable from your Word report via URLs like:

`http://127.0.0.1:8000/graphs/stage04_class_distribution.png`

## Startup notes

- First run downloads NLTK resources + the multilingual DistilBERT tokeniser
  (~1 minute of internet).
- The classifier and vectoriser are persisted in `artifacts/` after training,
  so subsequent starts skip re-training and `/classify` works immediately.
- No GPU needed. All endpoints run on CPU in seconds.

## Example call from Python

```python
import requests

r = requests.post("http://127.0.0.1:8000/classify",
                  json={"text": "How much severance am I owed after 6 years?"})
print(r.json())
# {'input': '...', 'intent': 'severance_query', 'confidence': 0.94,
#  'top_3': [['severance_query', 0.94], ['wage_query', 0.03], ...]}
```

## Where this fits in your report

Include a short subsection like this in your report:

> "The pipeline is deployed as a FastAPI service (`sheriabot NLP/api/`). Each
> NLP stage is exposed as a REST endpoint (`/stats`, `/nlp/pos`, `/nlp/ner`,
> `/nlp/topwords`, `/nlp/tfidf`, `/nlp/embedding`, `/nlp/tokenise`), and every
> graph produced by the pipeline is served as a static PNG under `/graphs`.
> The service reads the same Sheria-Bot Intents CSV that the notebook uses,
> so the results are identical."

Which mirrors exactly what the code does.
