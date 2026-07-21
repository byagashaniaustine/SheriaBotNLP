# Sheria-Bot WhatsApp API

FastAPI service that answers Tanzania employment-law questions over WhatsApp.
Runtime is intentionally narrow: **classify with BERT → generate answer → reply
via WhatsApp**. Training, EDA, and graph rendering all live outside `api/`.

## Layout

```
api/
├── main.py             ← FastAPI app: /health, /classify, /webhook
├── bert_classifier.py  ← wraps the fine-tuned DistilBERT model
├── answer_engine.py    ← intent + user text → localized reply (EN / SW)
├── whatsapp.py         ← Meta Cloud API integration (verify + send)
├── sessions.py         ← SQLite-backed per-user memory + retry dedup
├── schemas.py          ← Pydantic models
├── config.py           ← paths + API meta
├── requirements.txt    ← runtime deps only (fastapi, transformers, torch)
├── run.sh              ← one-command launcher
├── Procfile            ← Heroku/Railway
└── artifacts/
    ├── answer_bank.json    ← baked (intent, lang) → response  (required)
    └── sessions.db         ← created at first run
```

Related folders at the repo root (all independent of the runtime):

- `../sheriabot_bert_model/` — the **only** model the API loads (config.json,
  model.safetensors, tokenizer.json, vocab.txt, ...).
- `../data/` — raw CSVs used only for training in Colab. Runtime ignores them.
- `../graphs/` — PNGs produced by `SheriaBot_NLP_Graphs.py`. Runtime ignores them.

## Install & run

```bash
cd "sheriabot NLP/api"
pip install -r requirements.txt
uvicorn main:app --reload
```

Or:

```bash
bash run.sh
```

Swagger at **http://127.0.0.1:8000/docs**.

## Request flow

```
WhatsApp user
     │  text message
     ▼
Meta WhatsApp Cloud API  ──POST──▶  /webhook  (returns 200 immediately)
                                        │
                                        │  BackgroundTasks.add_task(...)
                                        ▼
                          bert_classifier.classify(text)
                                        │
                                        │  intent + confidence + top_3
                                        ▼
                          answer_engine.answer(text, intent, session)
                                        │
                                        │  answer text (EN or SW, personalized)
                                        ▼
                          whatsapp.send_text(sender, answer)
                                        │
                                        ▼
                          Meta WhatsApp Cloud API ──▶ user
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/`         | Service info / endpoint list |
| GET  | `/health`   | Model + answer-bank load state |
| POST | `/classify` | Debug: run the same classify+answer pipeline over HTTP |
| GET  | `/webhook`  | Meta verification handshake (`hub.mode`, `hub.verify_token`, `hub.challenge`) |
| POST | `/webhook`  | Meta inbound-message push |

Example:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Nimefutwa kazi bila notisi, nifanye nini?"}'
```

## Environment variables

| Var | Purpose |
|---|---|
| `WHATSAPP_VERIFY_TOKEN`    | Arbitrary string; must match Meta app dashboard |
| `WHATSAPP_ACCESS_TOKEN`    | Permanent (system-user) access token from Meta |
| `WHATSAPP_PHONE_NUMBER_ID` | WABA phone-number ID (numeric) |
| `WHATSAPP_GRAPH_VERSION`   | Optional, default `v20.0` |
| `HF_MODEL_ID`              | Optional; if set, load BERT from this HF Hub repo instead of `../sheriabot_bert_model/` |
| `SESSION_DB_PATH`          | Optional; default `artifacts/sessions.db`. Point at a persistent volume in production |

See `.env.example`.
