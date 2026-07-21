"""DistilBERT-based intent classifier — the only classifier the runtime uses.

Loads the fine-tuned model from `../sheriabot_bert_model/` (produced by the
Colab training notebook). The runtime never falls back to a classical model.

If `HF_MODEL_ID` is set, the model is pulled from that HuggingFace Hub repo
instead of the local directory — useful for Railway / Heroku deploys where you
don't want to ship the 500 MB safetensors in the git repo.
"""
from __future__ import annotations

import os
from typing import Dict, Any

from config import BERT_MODEL_DIR

HF_MODEL_ID_ENV = "HF_MODEL_ID"
MAX_LEN         = 96

_state: Dict[str, Any] = {"model": None, "tokenizer": None, "id2label": None,
                          "source": None}


def _local_model_ready() -> bool:
    return BERT_MODEL_DIR.exists() and (BERT_MODEL_DIR / "config.json").exists()


def _resolve_source() -> str:
    hub = os.getenv(HF_MODEL_ID_ENV)
    if hub:
        return hub
    if _local_model_ready():
        return str(BERT_MODEL_DIR)
    raise RuntimeError(
        f"No BERT model source: expected {BERT_MODEL_DIR} to contain "
        f"config.json + model.safetensors, or {HF_MODEL_ID_ENV} to point at "
        "a HuggingFace Hub repo id."
    )


def load() -> None:
    """Eager-load model + tokenizer once. Idempotent."""
    if _state["model"] is not None:
        return
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
    """Classify a single utterance. Returns intent + confidence + top-k."""
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


def is_loaded() -> bool:
    return _state["model"] is not None


def source() -> str:
    return _state["source"] or ""


def num_labels() -> int:
    return len(_state["id2label"] or {})
