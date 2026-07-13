"""
Standalone training script for SheriaBot intent classification.

Runs the full pipeline: load CSV, clean, TF-IDF, train 4 classifiers,
pick the best, save to disk. No GPU needed. Takes ~30 seconds on CPU.

Usage:
    cd "sheriabot NLP/api"
    python train.py

After training you can either use predict.py to classify new questions,
or start the FastAPI server and hit /classify.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


# ---------- paths ----------
HERE      = Path(__file__).resolve().parent
DATA_CSV  = HERE.parent / "data" / "04_Intents.csv"
ARTIFACTS = HERE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

RANDOM_SEED = 42


# ---------- 1. Load ----------
print("=" * 60)
print("STEP 1 / 5  -  Loading data")
print("=" * 60)
if not DATA_CSV.exists():
    raise FileNotFoundError(f"Cannot find {DATA_CSV}. Make sure this script "
                            f"lives in sheriabot NLP/api/ next to the data/ folder.")

df = pd.read_csv(DATA_CSV)
df = df.rename(columns={"utterance": "text", "intent_label": "intent"})
print(f"Loaded {len(df):,} rows, {df['intent'].nunique()} intent classes")


# ---------- 2. Clean ----------
print("\n" + "=" * 60)
print("STEP 2 / 5  -  Cleaning text")
print("=" * 60)

def clean_text(t: str) -> str:
    t = re.sub(r"\[.*?\]", " ", str(t))
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[0-9]+", " ", t.lower())
    t = re.sub(r"[^a-zÀ-ſ\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

n0 = len(df)
df["clean"] = df["text"].apply(clean_text)
df = df.dropna(subset=["text", "intent"]).drop_duplicates(subset=["clean"]).reset_index(drop=True)
print(f"Cleaned: {n0:,}  ->  {len(df):,} rows after removing duplicates & missing")


# ---------- 3. Features (TF-IDF) ----------
print("\n" + "=" * 60)
print("STEP 3 / 5  -  Feature engineering (TF-IDF)")
print("=" * 60)

vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X = vec.fit_transform(df["clean"])

le = LabelEncoder()
y = le.fit_transform(df["intent"])

print(f"TF-IDF matrix: {X.shape[0]:,} rows x {X.shape[1]:,} features")
print(f"Number of intent classes: {len(le.classes_)}")


# ---------- 4. Train four models ----------
print("\n" + "=" * 60)
print("STEP 4 / 5  -  Training four classifiers")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
print(f"Train: {X_train.shape[0]:,} rows   Test: {X_test.shape[0]:,} rows\n")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "NaiveBayes":         MultinomialNB(),
    "LinearSVM":          LinearSVC(),
    "RandomForest":       RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED
    ),
}

results = []
for name, m in models.items():
    print(f"  {name:20s} ", end="", flush=True)
    m.fit(X_train, y_train)
    train_acc = m.score(X_train, y_train)
    test_acc  = m.score(X_test, y_test)
    cv_mean   = float(cross_val_score(m, X, y, cv=5, n_jobs=-1).mean())
    print(f"train={train_acc:.3f}  test={test_acc:.3f}  cv={cv_mean:.3f}")
    results.append({"model": name, "train_acc": float(train_acc),
                    "test_acc": float(test_acc), "cv_mean": cv_mean})

results.sort(key=lambda r: r["test_acc"], reverse=True)
best_name = results[0]["model"]
best_model = models[best_name]

print(f"\nBest model: {best_name}   test accuracy = {results[0]['test_acc']:.4f}")


# ---------- 5. Evaluate + Save ----------
print("\n" + "=" * 60)
print("STEP 5 / 5  -  Evaluating winner and saving artifacts")
print("=" * 60)

y_pred = best_model.predict(X_test)
metrics = {
    "best_model":  best_name,
    "accuracy":    float(accuracy_score(y_test, y_pred)),
    "f1_macro":    float(f1_score(y_test, y_pred, average="macro")),
    "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    "results":     results,
}

print(f"\nAccuracy on test:   {metrics['accuracy']:.4f}")
print(f"F1 macro:           {metrics['f1_macro']:.4f}")
print(f"F1 weighted:        {metrics['f1_weighted']:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=le.classes_))

# save everything so /classify (and predict.py) can load them later
joblib.dump(best_model, ARTIFACTS / "classifier.joblib")
joblib.dump(vec,        ARTIFACTS / "tfidf_vec.joblib")
joblib.dump(le,         ARTIFACTS / "label_encoder.joblib")
(ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

# also drop a BoW vectoriser so /nlp/tfidf still works
bow = CountVectorizer(max_features=5000).fit(df["clean"])
joblib.dump(bow, ARTIFACTS / "bow_vec.joblib")

print(f"\nSaved to  {ARTIFACTS}")
for f in sorted(ARTIFACTS.iterdir()):
    print(f"  {f.name:30s} {f.stat().st_size // 1024:>6} KB")

print("\nDone. Try:  python predict.py \"Nimefutwa kazi bila notisi\"")
