"""Generate train_colab.ipynb from an in-file list of cells.

Rationale: assembling notebook JSON by hand is error-prone. This script keeps
the cell contents as ordinary Python strings, wraps them in the ipynb schema,
and writes a valid notebook file.

Regenerate with:  python build_colab_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

# ------------------------------------------------------------------
# Cell contents. (kind, source) tuples in display order.
# ------------------------------------------------------------------
CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Sheria-Bot — 5M dataset training (Colab)

This notebook trains the Sheria-Bot intent classifier on the 5-million-row dataset and produces the artifacts the FastAPI service needs to run on Railway.

**Inputs it expects on Google Drive:** `Sheria-Bot_5M_Dataset.zip` at `/content/drive/MyDrive/Sheria-Bot_5M_Dataset.zip` (or upload directly — see cell 2).

**Outputs it produces:** a downloadable `sheria_artifacts.zip` containing:
- `classifier.joblib`
- `tfidf_vec.joblib`
- `label_encoder.joblib`
- `answer_bank.json`
- `metrics.json`
- `eval_report.json`

Drop those into `sheriabot NLP/api/artifacts/` locally and redeploy.
"""),

    ("markdown", "## 1. Setup"),

    ("code", """\
# Colab already ships pandas/sklearn/joblib. Nothing extra to install.
import sys, platform
print("Python:", sys.version.split()[0], "on", platform.system())
"""),

    ("markdown", """\
## 2. Load the dataset

Two options — uncomment the one you're using.
"""),

    ("code", """\
# OPTION A — mount Google Drive (recommended: upload Sheria-Bot_5M_Dataset.zip to your Drive root first)
from google.colab import drive
drive.mount('/content/drive')

ZIP_PATH = '/content/drive/MyDrive/Sheria-Bot_5M_Dataset.zip'

# OPTION B — upload directly from your Mac (slower for 36 MB; skip if you did A)
# from google.colab import files
# uploaded = files.upload()          # pick Sheria-Bot_5M_Dataset.zip when prompted
# ZIP_PATH = next(iter(uploaded))
"""),

    ("code", """\
import zipfile, os
from pathlib import Path

EXTRACT_TO = Path('/content/sheria_5m')
if not EXTRACT_TO.exists():
    print(f'Extracting {ZIP_PATH} → {EXTRACT_TO} ...')
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_TO)

DATA_DIR = EXTRACT_TO / 'sheria_bot_5m_dataset' / 'data'
assert DATA_DIR.exists(), f'Extracted layout unexpected — {DATA_DIR} not found'

print('Files:')
for f in sorted(DATA_DIR.iterdir()):
    print(f'  {f.name:<40} {f.stat().st_size / 1_048_576:>7.1f} MB')
"""),

    ("markdown", """\
## 3. Load + clean the intents

The 5M `04_intents.csv` has 800 k rows. Loading takes ~15 s on Colab; cleaning ~30 s.
"""),

    ("code", """\
import re, pandas as pd, numpy as np, time

RANDOM_SEED = 42

_BR = re.compile(r'\\[.*?\\]')
_URL = re.compile(r'http\\S+|www\\S+')
_NUM = re.compile(r'[0-9]+')
_NON = re.compile(r'[^a-z\\u00c0-\\u017f\\s]')
_WS  = re.compile(r'\\s+')

def clean_text(t: str) -> str:
    t = _BR.sub(' ', str(t))
    t = _URL.sub(' ', t)
    t = _NUM.sub(' ', t.lower())
    t = _NON.sub(' ', t)
    return _WS.sub(' ', t).strip()

t0 = time.time()
df = pd.read_csv(DATA_DIR / '04_intents.csv')
print(f'Loaded {len(df):,} rows in {time.time()-t0:.1f}s')
df = df.rename(columns={'utterance': 'text', 'intent_label': 'intent'})
df['clean'] = df['text'].astype(str).map(clean_text)
df = df.dropna(subset=['text', 'intent']).drop_duplicates(subset=['clean']).reset_index(drop=True)
print(f'After clean + dedupe: {len(df):,} rows, {df["intent"].nunique()} intents, langs = {df["lang"].value_counts().to_dict()}')
"""),

    ("markdown", """\
## 4. Feature engineering (TF-IDF)

Bumped from the small-dataset config (`max_features=5000`, `min_df=2`) to sizes that fit an 800 k-row vocabulary. Adjust down if Colab RAM runs tight.
"""),

    ("code", """\
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

TFIDF_MAX_FEATURES = 50_000
TFIDF_MIN_DF       = 5
TFIDF_NGRAM        = (1, 2)

t0 = time.time()
vec = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM,
    min_df=TFIDF_MIN_DF,
    sublinear_tf=True,
)
X = vec.fit_transform(df['clean'])
le = LabelEncoder()
y = le.fit_transform(df['intent'])

print(f'TF-IDF matrix: {X.shape[0]:,} × {X.shape[1]:,}  (nnz = {X.nnz:,})  in {time.time()-t0:.1f}s')
print(f'Classes: {len(le.classes_)}')
"""),

    ("markdown", """\
## 5. Train two classifiers

RandomForest and MultinomialNB are skipped — RF is slow at 800 k, NB caps low. LogisticRegression (SAGA solver) + LinearSVM cover the accuracy space.
"""),

    ("code", """\
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y,
)
print(f'Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}')

models = {
    'LogisticRegression': LogisticRegression(
        solver='saga', max_iter=200, n_jobs=-1, C=1.0,
    ),
    'LinearSVM': LinearSVC(C=1.0),
}

results = []
for name, m in models.items():
    t0 = time.time()
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    f1w = f1_score(y_test, y_pred, average='weighted')
    print(f'{name:22s}  test_acc={acc:.4f}  f1_macro={f1m:.4f}  f1_weighted={f1w:.4f}  ({time.time()-t0:.1f}s)')
    results.append({'model': name, 'accuracy': float(acc), 'f1_macro': float(f1m), 'f1_weighted': float(f1w)})

results.sort(key=lambda r: r['accuracy'], reverse=True)
best_name = results[0]['model']
best_model = models[best_name]
print(f'\\nBest: {best_name} @ {results[0]["accuracy"]:.4f}')
"""),

    ("markdown", """\
## 6. Save classifier artifacts
"""),

    ("code", """\
import joblib, json
from pathlib import Path

ARTIFACTS = Path('/content/artifacts')
ARTIFACTS.mkdir(exist_ok=True)

joblib.dump(best_model, ARTIFACTS / 'classifier.joblib')
joblib.dump(vec,        ARTIFACTS / 'tfidf_vec.joblib')
joblib.dump(le,         ARTIFACTS / 'label_encoder.joblib')

metrics = {
    'best_model':  best_name,
    'accuracy':    results[0]['accuracy'],
    'f1_macro':    results[0]['f1_macro'],
    'f1_weighted': results[0]['f1_weighted'],
    'results':     results,
    'train_rows':  int(X_train.shape[0]),
    'test_rows':   int(X_test.shape[0]),
    'features':    int(X.shape[1]),
    'intents':     int(len(le.classes_)),
}
(ARTIFACTS / 'metrics.json').write_text(json.dumps(metrics, indent=2))
print('Saved:', sorted(p.name for p in ARTIFACTS.iterdir()))
"""),

    ("markdown", """\
## 7. Bake `answer_bank.json` from the NLU mapping

400 k rows collapse to ~62 unique responses (one per intent × language). This is the file the FastAPI answer engine reads at runtime.
"""),

    ("code", """\
nlu = pd.read_csv(DATA_DIR / '05_nlu_knowledge_mapping.csv')
print(f'NLU mapping: {len(nlu):,} rows')

bank: dict = {}
for (intent, lang), grp in nlu.groupby(['parsed_intent', 'lang']):
    # Most-common solution_response for this (intent, lang), not just the first —
    # helps when sector/region variants edited the boilerplate slightly.
    best_response = grp['solution_response'].astype(str).str.strip().value_counts().idxmax()
    row = grp[grp['solution_response'] == best_response].iloc[0]
    bank.setdefault(intent, {})[lang] = {
        'response': best_response,
        'citation': str(row.get('citation', '') or '').strip(),
    }

(ARTIFACTS / 'answer_bank.json').write_text(json.dumps(bank, ensure_ascii=False, indent=2))
print(f'Baked {sum(len(v) for v in bank.values())} entries across {len(bank)} intents')
"""),

    ("markdown", """\
## 8. Evaluate on `10_evaluation.csv`

This is the honest test: for each held-out question, classify → look up the KB answer → check whether it contains the `expected_answer_substring`. That measures *answer* quality, not just intent accuracy.
"""),

    ("code", """\
ev = pd.read_csv(DATA_DIR / '10_evaluation.csv')
print(f'Eval set: {len(ev):,} rows')
print('Difficulty:', ev['difficulty'].value_counts().to_dict() if 'difficulty' in ev.columns else 'n/a')

def language_of(text: str) -> str:
    SW = {'na','ya','wa','ni','kwa','za','la','ku','katika','nimefutwa','nifanye',
          'sijui','tafadhali','asante','habari','mimi','kazi','mshahara','likizo'}
    toks = re.findall(r'[a-z\\u00c0-\\u017f]+', str(text).lower())
    return 'sw' if any(t in SW for t in toks) else 'en'

def predict_intent(question: str) -> str:
    x = vec.transform([clean_text(question)])
    return le.classes_[best_model.predict(x)[0]]

def lookup_answer(intent: str, lang: str) -> str:
    return (bank.get(intent, {}).get(lang) or bank.get(intent, {}).get('en', {})).get('response', '')

t0 = time.time()
n_intent_ok = 0
n_answer_ok = 0
n_citation_ok = 0
mistakes = []

for _, row in ev.iterrows():
    q = row['question']
    expected_substr = str(row.get('expected_answer_substring', '') or '').strip()
    expected_citation = str(row.get('expected_citation', '') or '').strip()
    lang = row.get('lang') or language_of(q)

    intent = predict_intent(q)
    ans = lookup_answer(intent, lang)

    intent_ok = intent in q.lower() or expected_substr.lower() in ans.lower()
    ans_ok = bool(expected_substr) and expected_substr.lower() in ans.lower()
    cit_ok = bool(expected_citation) and expected_citation.lower() in ans.lower()

    if ans_ok:
        n_answer_ok += 1
    if cit_ok:
        n_citation_ok += 1
    if not ans_ok and len(mistakes) < 15:
        mistakes.append({
            'q': q[:80],
            'expected': expected_substr[:80],
            'got_intent': intent,
            'got_ans': ans[:80],
        })

total = len(ev)
print(f'\\nEvaluated {total:,} questions in {time.time()-t0:.1f}s')
print(f'  answer contains expected substring : {n_answer_ok:,} / {total:,} = {n_answer_ok/total:.3%}')
print(f'  answer contains expected citation  : {n_citation_ok:,} / {total:,} = {n_citation_ok/total:.3%}')

print('\\nFirst mistakes:')
for m in mistakes:
    print(f'  Q: {m["q"]}')
    print(f'    expected substr: {m["expected"]!r}')
    print(f'    got intent={m["got_intent"]}, ans starts: {m["got_ans"]!r}')

eval_report = {
    'total':            int(total),
    'answer_match':     int(n_answer_ok),
    'answer_match_pct': n_answer_ok / total,
    'citation_match':   int(n_citation_ok),
    'citation_match_pct': n_citation_ok / total,
}
(ARTIFACTS / 'eval_report.json').write_text(json.dumps(eval_report, indent=2))
"""),

    ("markdown", """\
## 9. Package + persist + download

Colab's `/content/` is wiped when the runtime disconnects, so we do three things:

1. Copy `/content/artifacts/` to Google Drive at `MyDrive/sheria_artifacts/` — survives runtime restarts, and next time you can skip retraining and just pull them back.
2. Also save a zipped copy at `MyDrive/sheria_artifacts.zip`.
3. Trigger a browser download of the zip so you can drop it into the repo locally.
"""),

    ("code", """\
import shutil
from pathlib import Path

DRIVE_ROOT   = Path('/content/drive/MyDrive')
DRIVE_DIR    = DRIVE_ROOT / 'sheria_artifacts'
DRIVE_ZIP    = DRIVE_ROOT / 'sheria_artifacts.zip'
LOCAL_ZIP    = '/content/sheria_artifacts'   # .zip added by make_archive

# 1. Persist unzipped folder to Drive (overwrite any prior run)
if DRIVE_DIR.exists():
    shutil.rmtree(DRIVE_DIR)
shutil.copytree(ARTIFACTS, DRIVE_DIR)
print(f'Copied artifacts → {DRIVE_DIR}')

# 2. Also persist a zip to Drive (for easy download later without re-training)
shutil.make_archive(LOCAL_ZIP, 'zip', ARTIFACTS)
shutil.copy(f'{LOCAL_ZIP}.zip', DRIVE_ZIP)
print(f'Copied zip      → {DRIVE_ZIP}')

print('\\nContents:')
for p in sorted(ARTIFACTS.iterdir()):
    print(f'  {p.name:30s} {p.stat().st_size / 1024:>8.1f} KB')

# 3. Trigger browser download to your Mac
from google.colab import files
files.download(f'{LOCAL_ZIP}.zip')
"""),

    ("markdown", """\
## 10. Install locally

On your Mac, from the project root:

```bash
cd "sheriabot NLP/api"
rm -f artifacts/*.joblib artifacts/*.json    # remove the previous artifacts
unzip -o ~/Downloads/sheria_artifacts.zip -d artifacts/
ls -la artifacts/
git add artifacts/
git commit -m "Retrain on 5M dataset"
git push
```

Railway auto-redeploys from `git push`. The bot restarts with the new model.

## Skip re-training next time — pull from Drive

Because step 9 above also wrote everything to `MyDrive/sheria_artifacts/` and
`MyDrive/sheria_artifacts.zip`, you don't have to re-run the notebook to
re-download the artifacts. Any of these works:

- From your Mac: right-click `sheria_artifacts.zip` in Google Drive web →
  Download, then unzip into `api/artifacts/`.
- From a fresh Colab session: `!cp /content/drive/MyDrive/sheria_artifacts.zip .`
  after mounting Drive.
"""),
]


# ------------------------------------------------------------------
# Assemble ipynb
# ------------------------------------------------------------------
def to_source(s: str) -> list[str]:
    """Notebook cell source is stored as a list of lines, each ending in \\n
    except the last. This matches the standard ipynb convention."""
    if not s:
        return []
    lines = s.splitlines(keepends=True)
    # The last line usually needs no trailing newline to match conventions.
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines


def build_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for kind, src in cells:
        cell: dict = {
            "cell_type": kind,
            "metadata": {},
            "source": to_source(src),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language":     "python",
                "name":         "python3",
            },
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "train_colab.ipynb"
    nb = build_notebook(CELLS)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Wrote {out}  ({out.stat().st_size / 1024:.1f} KB, {len(CELLS)} cells)")
