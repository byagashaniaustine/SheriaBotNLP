# Training data (not in git)

The training CSVs (5M-row Sheria-Bot dataset) are **intentionally not committed**
to this repository. Reasons:

- Individual files exceed GitHub's 100 MB per-file cap
  (`04_intents.csv` ≈ 127 MB, `05_nlu_knowledge_mapping.csv` ≈ 155 MB)
- The runtime bot does not need them — it reads only `../api/artifacts/`
  (see the "runtime is artifacts-only" principle below)
- Training happens in Google Colab against the raw dataset; the resulting
  artifacts (`classifier.joblib`, `tfidf_vec.joblib`, `label_encoder.joblib`,
  `answer_bank.json`, `metrics.json`) are what get deployed.

## How to obtain the dataset

1. Download `Sheria-Bot_5M_Dataset.zip` from your Google Drive (or wherever
   the maintainer publishes it).
2. Unzip so this `data/` folder contains the 10 files:

   ```
   01_vocabulary.csv
   02_legal_terms.csv
   03_introductions.csv
   04_intents.csv
   05_nlu_knowledge_mapping.csv
   06_semantics.csv
   07_dialogues.jsonl
   08_contracts.csv
   09_breaches.csv
   10_evaluation.csv
   ```

3. The `.gitignore` at the repo root will keep them out of git automatically.

## Runtime = artifacts-only

The FastAPI service in `../api/main.py` **never opens a CSV at runtime**:

| Runtime file | Purpose |
|---|---|
| `api/artifacts/classifier.joblib` | trained sklearn classifier |
| `api/artifacts/tfidf_vec.joblib` | fitted TF-IDF vectorizer |
| `api/artifacts/label_encoder.joblib` | intent label ↔ integer |
| `api/artifacts/answer_bank.json` | baked legal answers by (intent, lang) |
| `api/artifacts/sessions.db` | SQLite session store (created at first run) |

The CSVs in this folder are read only by:
- `api/train.py` (local training on the small dataset)
- `api/bake_answers.py` (local baking of `answer_bank.json`)
- `api/train_colab.ipynb` (Colab training — expected workflow)
- The `/stats`, `/nlp/pos`, `/nlp/ner`, `/nlp/topwords` analytical endpoints
  (dev-only; return 503 if `data/` is empty)
