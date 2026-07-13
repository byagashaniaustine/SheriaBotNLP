"""
Standalone prediction script for SheriaBot intent classifier.

Loads the trained model from artifacts/ and classifies a new question.

Usage:
    python predict.py "Nimefutwa kazi bila notisi, nifanye nini?"
    python predict.py "How much severance am I owed after 6 years?"
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np

HERE      = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def clean_text(t: str) -> str:
    t = re.sub(r"\[.*?\]", " ", str(t))
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[0-9]+", " ", t.lower())
    t = re.sub(r"[^a-zÀ-ſ\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def classify(text: str, top_k: int = 3):
    if not (ARTIFACTS / "classifier.joblib").exists():
        print("ERROR: no trained model found. Run  python train.py  first.")
        sys.exit(1)

    model = joblib.load(ARTIFACTS / "classifier.joblib")
    vec   = joblib.load(ARTIFACTS / "tfidf_vec.joblib")
    le    = joblib.load(ARTIFACTS / "label_encoder.joblib")

    x = vec.transform([clean_text(text)])
    pred = model.predict(x)[0]
    intent = le.classes_[pred]

    result = {"input": text, "intent": intent}

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        top = np.argsort(probs)[::-1][:top_k]
        result["confidence"] = float(probs[pred])
        result["top_3"] = [(le.classes_[i], float(probs[i])) for i in top]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x)[0]
        top = np.argsort(scores)[::-1][:top_k]
        result["top_3"] = [(le.classes_[i], float(scores[i])) for i in top]

    return result


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    text = " ".join(sys.argv[1:])
    result = classify(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
