"""Optional DistilBERT-based intent classifier.

Activation rules (checked at import time by is_available()):
  1. If artifacts/bert_model/ exists locally with a config.json — use it.
  2. Else if env var HF_MODEL_ID is set — pull that repo from HuggingFace Hub
     on first call (transformers caches it under ~/.cache/huggingface/hub).
  3. Otherwise: unavailable, and pipeline.classify_text falls back to sklearn.

The runtime never depends on training-time CSVs. It reads:
  - artifacts/bert_model/{config.json, model.safetensors, tokenizer.json, ...}
  - or a HuggingFace Hub repo (downloaded once, cached on disk / in a Railway
    volume mounted at ~/.cache/huggingface).

Kept lazy so the sklearn path doesn't pay import cost for torch when BERT
isn't in use.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

from config import ARTIFACTS

BERT_MODEL_DIR    = ARTIFACTS / "bert_model"
HF_MODEL_ID_ENV   = "HF_MODEL_ID"
MAX_LEN           = 96

_state: Dict[str, Any] = {"model": None, "tokenizer": None, "id2label": None,
                         "source": None}


def _local_model_ready() -> bool:
    return BERT_MODEL_DIR.exists() and (BERT_MODEL_DIR / "config.json").exists()


def is_available() -> bool:
    """True if we can plausibly load a BERT model — from disk or HF Hub."""
    return _local_model_ready() or bool(os.getenv(HF_MODEL_ID_ENV))


def _resolve_source() -> str:
    if _local_model_ready():
        return str(BERT_MODEL_DIR)
    hub = os.getenv(HF_MODEL_ID_ENV)
    if hub:
        return hub
    raise RuntimeError(
        f"No BERT model source: expected {BERT_MODEL_DIR} to contain a trained "
        f"model, or {HF_MODEL_ID_ENV} env var set to a HuggingFace Hub repo id."
    )


def load() -> None:
    """Eager-load the model + tokenizer. Idempotent."""
    if _state["model"] is not None:
        return
    # Deferred import — installing torch is expensive and only needed if BERT
    # is actually being used.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    src = _resolve_source()
    print(f"[bert] loading from {src}")
    _state["tokenizer"] = AutoTokenizer.from_pretrained(src)
    m = AutoModelForSequenceClassification.from_pretrained(src)
    m.eval()
    _state["model"] = m
    _state["id2label"] = {int(k): v for k, v in m.config.id2label.items()}
    _state["source"] = src
    print(f"[bert] ready — {len(_state['id2label'])} labels")


def classify(text: str, top_k: int = 3) -> Dict[str, Any]:
    """Classify a single utterance. Same return shape as pipeline.classify_text
    so the webhook code doesn't care which backend produced it."""
    import torch
    load()
    tok      = _state["tokenizer"]
    m        = _state["model"]
    id2label = _state["id2label"]

    inputs = tok(text, return_tensors="pt", truncation=True,
                 padding="max_length", max_length=MAX_LEN)
    with torch.no_grad():
        logits = m(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    top = torch.argsort(probs, descending=True)[:top_k].tolist()

    return {
        "input":      text,
        "intent":     id2label[top[0]],
        "confidence": float(probs[top[0]]),
        "top_3":      [(id2label[i], float(probs[i])) for i in top],
    }


def source() -> str:
    """Returns the resolved source path/HF id after load(). '' if not loaded."""
    return _state["source"] or ""
