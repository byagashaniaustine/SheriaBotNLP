"""Sheria-Bot NLP pipeline — pure Python, no FastAPI dependencies.

This module exposes callable functions for each NLP stage.
`main.py` (FastAPI) uses these functions behind HTTP endpoints.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd

# ---- NLTK ----
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

# ---- Scikit-learn ----
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ---- Gensim (Word2Vec) ----
from gensim.models import Word2Vec

from config import (
    INTENTS_CSV, ARTIFACTS,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF,
    TEST_SIZE, RANDOM_SEED, CV_FOLDS,
    TRANSFORMER_MODEL,
)

# --------------------------------------------------------------------------
# NLTK resources
# --------------------------------------------------------------------------
def ensure_nltk() -> None:
    for r in ["punkt", "punkt_tab", "stopwords", "wordnet",
              "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
              "maxent_ne_chunker", "maxent_ne_chunker_tab", "words", "omw-1.4"]:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Bilingual stop-word set
# --------------------------------------------------------------------------
def stopwords_bilingual() -> set:
    en = set(stopwords.words("english"))
    sw = {"na","ya","wa","ni","kwa","za","la","ku","katika","hii","hiyo","kama",
          "tu","pia","yangu","yako","yake","yenu","yao","ndani","nje","juu","chini",
          "hapa","huko","huku","kabisa","tayari","bado","siku","mwezi","mwaka",
          "mimi","wewe","yeye","sisi","nyinyi","wao","mtu","watu","kitu","vitu",
          "je","nini","gani","lini","wapi","vipi","kiasi","ngapi","nani"}
    return en | sw


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
_BRACKETS = re.compile(r"\[.*?\]")
_URL      = re.compile(r"http\S+|www\S+")
_NUMBERS  = re.compile(r"[0-9]+")
_NONLET   = re.compile(r"[^a-zÀ-ſ\s]")
_SPACES   = re.compile(r"\s+")


def clean_text(t: str) -> str:
    t = _BRACKETS.sub(" ", str(t))
    t = _URL.sub(" ", t)
    t = _NUMBERS.sub(" ", t.lower())
    t = _NONLET.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


# --------------------------------------------------------------------------
# In-memory state (populated by load_state at startup)
# --------------------------------------------------------------------------
STATE: Dict[str, Any] = {
    "df": None,
    "STOP": None,
    "bow_vec": None,
    "tfidf_vec": None,
    "X_tfidf": None,
    "y_enc": None,
    "label_encoder": None,
    "w2v": None,
    "classifier": None,
    "hf_tokenizer": None,
    "metrics": None,
}


EDA_MAX_ROWS = 20_000       # cap for in-process EDA on the 800k-row 5M CSV
LARGE_FILE_MB = 20          # anything bigger gets sampled instead of fully loaded


def load_state() -> Dict[str, Any]:
    """Load dataset, build stopwords, and return state dict.

    For the 5M dataset the intents file is ~127 MB / 800 k rows — loading the
    whole thing at startup costs ~30 s and ~2 GB RAM per FastAPI worker, which
    isn't tenable on Railway. When the file is large we sample EDA_MAX_ROWS
    stratified across intents; the webhook doesn't use STATE["df"] anyway
    (it reads the joblib-persisted model), so the sample only affects the
    /stats and /nlp/* analytical endpoints.
    """
    ensure_nltk()
    file_size_mb = INTENTS_CSV.stat().st_size / (1024 * 1024)
    if file_size_mb > LARGE_FILE_MB:
        print(f"[pipeline] {INTENTS_CSV.name} is {file_size_mb:.0f} MB — "
              f"loading a {EDA_MAX_ROWS:,}-row sample for EDA endpoints.")
        df = pd.read_csv(INTENTS_CSV, nrows=EDA_MAX_ROWS)
    else:
        df = pd.read_csv(INTENTS_CSV)

    df = df.rename(columns={"utterance": "text", "intent_label": "intent"})
    df["length"] = df["text"].astype(str).apply(len)
    df["clean"]  = df["text"].apply(clean_text)
    df = df.dropna(subset=["text", "intent"]).drop_duplicates(subset=["clean"]).reset_index(drop=True)

    STATE["df"] = df
    STATE["STOP"] = stopwords_bilingual()
    print(f"[pipeline] loaded {len(df)} rows, {df['intent'].nunique()} intents")
    return STATE


# --------------------------------------------------------------------------
# Stage 4 — EDA
# --------------------------------------------------------------------------
def eda_stats() -> Dict[str, Any]:
    df = STATE["df"]
    return {
        "total_rows":     int(len(df)),
        "unique_intents": int(df["intent"].nunique()),
        "intent_counts":  df["intent"].value_counts().to_dict(),
        "language_counts": df["lang"].value_counts().to_dict(),
        "avg_length_chars": float(df["length"].mean()),
    }


# --------------------------------------------------------------------------
# Stage 11 — POS tagging
# --------------------------------------------------------------------------
_POS_MEANING = {
    "NN":"noun (singular)", "NNS":"noun (plural)", "NNP":"proper noun",
    "JJ":"adjective", "VB":"verb (base)", "VBD":"verb (past)", "VBG":"verb (-ing)",
    "VBN":"verb (past participle)", "VBP":"verb (present)", "VBZ":"verb (3rd sg)",
    "RB":"adverb", "IN":"preposition", "DT":"determiner", "PRP":"pronoun",
    "CC":"conjunction", "CD":"cardinal number", "TO":"to", "MD":"modal",
    "WP":"wh-pronoun", "WRB":"wh-adverb", "WDT":"wh-determiner",
}


def pos_distribution(sample: int = 300, top_n: int = 15) -> Dict[str, Any]:
    df = STATE["df"]
    tags = []
    for text in df["clean"].sample(min(sample, len(df)), random_state=RANDOM_SEED):
        tags.extend(t for _, t in pos_tag(word_tokenize(text)))
    top = pd.Series(tags).value_counts().head(top_n).to_dict()
    meanings = {t: _POS_MEANING.get(t, "?") for t in top}
    return {"sample_size": sample, "top_tags": top, "tag_meanings": meanings}


# --------------------------------------------------------------------------
# Stage 12 — NER
# --------------------------------------------------------------------------
def ner_counts(sample: int = 400) -> Dict[str, Any]:
    df = STATE["df"]
    counts: Counter = Counter()
    examples: Dict[str, str] = {}
    for text in df["text"].sample(min(sample, len(df)), random_state=RANDOM_SEED):
        try:
            tree = ne_chunk(pos_tag(word_tokenize(str(text))))
        except Exception:
            continue
        for sub in tree:
            if hasattr(sub, "label"):
                lbl = sub.label()
                counts[lbl] += 1
                if lbl not in examples:
                    examples[lbl] = " ".join(w for w, _ in sub.leaves())
    return {"sample_size": sample,
            "entity_counts": dict(counts),
            "entity_examples": examples}


# --------------------------------------------------------------------------
# Stage 13 — top words
# --------------------------------------------------------------------------
def top_words(n: int = 20) -> Dict[str, Any]:
    df = STATE["df"]
    STOP = STATE["STOP"]
    all_tokens = []
    for txt in df["clean"]:
        all_tokens.extend(w for w in word_tokenize(txt) if w.isalpha() and w not in STOP)
    freq = Counter(all_tokens)
    return {"top_words": freq.most_common(n),
            "total_tokens": len(all_tokens)}


# --------------------------------------------------------------------------
# Stage 14 — feature engineering (BoW + TF-IDF)
# --------------------------------------------------------------------------
def build_features() -> Dict[str, Any]:
    df = STATE["df"]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean"], df["intent"], test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=df["intent"]
    )
    bow_vec = CountVectorizer(max_features=TFIDF_MAX_FEATURES)
    bow_vec.fit(X_train_text)

    tfidf_vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                                ngram_range=TFIDF_NGRAM_RANGE,
                                min_df=TFIDF_MIN_DF)
    X_tfidf = tfidf_vec.fit_transform(df["clean"])

    le = LabelEncoder()
    y_enc = le.fit_transform(df["intent"])

    STATE["bow_vec"]       = bow_vec
    STATE["tfidf_vec"]     = tfidf_vec
    STATE["X_tfidf"]       = X_tfidf
    STATE["y_enc"]         = y_enc
    STATE["label_encoder"] = le

    tfidf_sum = np.asarray(X_tfidf.sum(axis=0)).ravel()
    top_idx   = tfidf_sum.argsort()[::-1][:20]
    top_terms = [(str(t), float(s)) for t, s in
                 zip(np.array(tfidf_vec.get_feature_names_out())[top_idx],
                     tfidf_sum[top_idx])]
    return {
        "top_terms": top_terms,
        "vocab_size_bow":   len(bow_vec.get_feature_names_out()),
        "vocab_size_tfidf": len(tfidf_vec.get_feature_names_out()),
    }


# --------------------------------------------------------------------------
# Stage 15 — Word2Vec embeddings
# --------------------------------------------------------------------------
def build_word2vec(vector_size: int = 100, epochs: int = 15) -> None:
    df = STATE["df"]
    tokenised = [word_tokenize(t) for t in df["clean"]]
    w2v = Word2Vec(sentences=tokenised, vector_size=vector_size, window=5,
                   min_count=5, workers=1, seed=RANDOM_SEED, epochs=epochs)
    STATE["w2v"] = w2v
    print(f"[pipeline] Word2Vec trained: {len(w2v.wv)} vocabulary")


def embedding_for(word: str, top_k: int = 5) -> Dict[str, Any]:
    w2v = STATE["w2v"]
    if w2v is None:
        raise RuntimeError("Word2Vec not trained yet — call /pipeline/train first.")
    if word not in w2v.wv:
        return {"word": word, "vector_preview": [], "vector_size": 0,
                "similar_words": []}
    vec = w2v.wv[word]
    sim = w2v.wv.most_similar(word, topn=top_k)
    return {
        "word": word,
        "vector_preview": [float(x) for x in vec[:10]],
        "vector_size": int(vec.shape[0]),
        "similar_words": [(w, float(s)) for w, s in sim],
    }


# --------------------------------------------------------------------------
# Stage 16 — transformer tokenisation
# --------------------------------------------------------------------------
def get_hf_tokenizer():
    if STATE["hf_tokenizer"] is None:
        from transformers import AutoTokenizer
        STATE["hf_tokenizer"] = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    return STATE["hf_tokenizer"]


def tokenise_compare(text: str) -> Dict[str, Any]:
    tok = get_hf_tokenizer()
    cleaned = clean_text(text)
    nltk_tokens = word_tokenize(cleaned)
    bert_ids    = tok(cleaned)["input_ids"]
    bert_subs   = tok.convert_ids_to_tokens(bert_ids)
    n_nltk = len(nltk_tokens)
    n_bert = len(bert_subs)
    return {
        "input": text,
        "nltk_tokens": nltk_tokens,
        "bert_subwords": bert_subs,
        "nltk_count": n_nltk,
        "bert_count": n_bert,
        "ratio": (n_bert / n_nltk) if n_nltk else 0.0,
    }


# --------------------------------------------------------------------------
# Training + inference
# --------------------------------------------------------------------------
def train_classifiers() -> Dict[str, Any]:
    """Train four classifiers, pick the best, and save to disk."""
    X = STATE["X_tfidf"]
    y = STATE["y_enc"]
    if X is None:
        raise RuntimeError("Features not built — call /pipeline/features first.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    models: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "NaiveBayes":         MultinomialNB(),
        "LinearSVM":          LinearSVC(),
        "RandomForest":       RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED
        ),
    }

    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        train_acc = m.score(X_train, y_train)
        test_acc  = m.score(X_test, y_test)
        cv_mean   = float(cross_val_score(m, X, y, cv=CV_FOLDS, n_jobs=-1).mean())
        results.append({"model": name, "train_acc": float(train_acc),
                        "test_acc": float(test_acc), "cv_mean": cv_mean})

    results.sort(key=lambda r: r["test_acc"], reverse=True)
    best = results[0]
    best_model = models[best["model"]]

    y_pred = best_model.predict(X_test)
    metrics = {
        "best_model":  best["model"],
        "accuracy":    float(accuracy_score(y_test, y_pred)),
        "f1_macro":    float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "results":     results,
    }

    STATE["classifier"] = best_model
    STATE["metrics"]    = metrics

    joblib.dump(best_model,          ARTIFACTS / "classifier.joblib")
    joblib.dump(STATE["tfidf_vec"],  ARTIFACTS / "tfidf_vec.joblib")
    joblib.dump(STATE["label_encoder"], ARTIFACTS / "label_encoder.joblib")
    (ARTIFACTS / "metrics.json").write_text(str(metrics))

    return metrics


def try_load_saved_classifier() -> bool:
    """Load model+vectoriser+encoder from disk if a previous run trained them.

    Two backends supported. Precedence:
      1. DistilBERT (if artifacts/bert_model/ or HF_MODEL_ID env var present)
      2. sklearn TF-IDF (if artifacts/classifier.joblib present)

    The chosen backend is recorded in STATE["backend"] and used by classify_text().
    """
    import bert_classifier
    if bert_classifier.is_available():
        try:
            bert_classifier.load()
            STATE["backend"] = "bert"
            print(f"[pipeline] classifier backend: BERT ({bert_classifier.source()})")
            return True
        except Exception as e:
            print(f"[pipeline] BERT load failed ({e!r}) — falling back to sklearn")

    p = ARTIFACTS / "classifier.joblib"
    if not p.exists():
        STATE["backend"] = None
        return False
    STATE["classifier"]    = joblib.load(p)
    STATE["tfidf_vec"]     = joblib.load(ARTIFACTS / "tfidf_vec.joblib")
    STATE["label_encoder"] = joblib.load(ARTIFACTS / "label_encoder.joblib")
    STATE["backend"] = "sklearn"
    print(f"[pipeline] classifier backend: sklearn ({p.stat().st_size // 1024} KB model)")
    return True


def classify_text(text: str, top_k: int = 3) -> Dict[str, Any]:
    """Classify a single legal question and return the top-k intents.
    Dispatches to whichever backend was loaded by try_load_saved_classifier()."""
    if STATE.get("backend") == "bert":
        import bert_classifier
        return bert_classifier.classify(text, top_k=top_k)

    clf = STATE["classifier"]
    vec = STATE["tfidf_vec"]
    le  = STATE["label_encoder"]
    if clf is None:
        raise RuntimeError("Classifier not trained — call /train first.")

    cleaned = clean_text(text)
    x = vec.transform([cleaned])
    pred = clf.predict(x)[0]
    intent = le.classes_[pred]

    result: Dict[str, Any] = {
        "input": text, "intent": intent, "confidence": None, "top_3": [],
    }
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(x)[0]
        top_idx = np.argsort(p)[::-1][:top_k]
        result["confidence"] = float(p[pred])
        result["top_3"] = [(le.classes_[i], float(p[i])) for i in top_idx]
    elif hasattr(clf, "decision_function"):
        s = clf.decision_function(x)[0]
        top_idx = np.argsort(s)[::-1][:top_k]
        result["top_3"] = [(le.classes_[i], float(s[i])) for i in top_idx]
    return result
