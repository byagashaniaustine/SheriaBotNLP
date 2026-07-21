"""Semantic retrieval over SheriaBot's law knowledge base.

At runtime this module owns TWO artifacts produced once by build_rag_index.py:

    api/artifacts/rag_index.npz     — float32 array [n_chunks, embedding_dim]
    api/artifacts/rag_chunks.json   — list of chunk dicts, aligned to the array

Query flow:

    1. embed the user's question with the same sentence-transformer that
       produced the index (paraphrase-multilingual-MiniLM-L12-v2, 118 MB)
    2. cosine-similarity against every chunk vector
    3. return the top-k chunks (text + citation + score)

The retriever loads lazily on first call, so importing this module never
takes more than a few milliseconds — the ~118 MB sentence-transformer only
paid for when retrieve() is first invoked.

If either artifact is missing (or sentence-transformers isn't installed),
retrieve() returns [] and the caller (answer_engine) falls back to the
canned answer bank. This is the graceful-degradation contract.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import ARTIFACTS

log = logging.getLogger("sheriabot.rag")

INDEX_NPZ    = ARTIFACTS / "rag_index.npz"
CHUNKS_JSON  = ARTIFACTS / "rag_chunks.json"

# Model name. Kept in sync with build_rag_index.py — if you change one you
# MUST rebuild the other, otherwise vectors won't match dimensionally.
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class _Retriever:
    """Owns the sentence-transformer + the chunk store. Singleton pattern."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._embeddings: Optional[np.ndarray] = None      # [n, d], L2-normalised
        self._chunks: List[Dict[str, Any]] = []
        self._ready: bool = False
        self._load_index()

    # ------------------------------------------------------------------
    # index loading — happens at import time, cheap (JSON + npz only)
    # ------------------------------------------------------------------
    def _load_index(self) -> None:
        if not INDEX_NPZ.exists() or not CHUNKS_JSON.exists():
            log.warning(
                "RAG artifacts missing (expected %s and %s). Retrieval will "
                "return empty; answer engine will fall back to answer_bank. "
                "Run build_rag_index.py to create them.",
                INDEX_NPZ, CHUNKS_JSON,
            )
            return
        try:
            arr = np.load(INDEX_NPZ)
            self._embeddings = arr["embeddings"].astype(np.float32)
            self._chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
        except Exception as e:
            log.error("Failed to load RAG index: %s", e)
            return

        if len(self._chunks) != self._embeddings.shape[0]:
            log.error(
                "RAG index mismatch: %d chunks vs %d embedding rows. Rebuild.",
                len(self._chunks), self._embeddings.shape[0],
            )
            return
        self._ready = True
        log.info(
            "RAG ready: %d chunks, embedding dim=%d",
            len(self._chunks), self._embeddings.shape[1],
        )

    # ------------------------------------------------------------------
    # sentence-transformer — expensive, loaded lazily on first retrieve()
    # ------------------------------------------------------------------
    def _load_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.warning(
                "sentence-transformers not installed. RAG disabled. "
                "Install with: pip install sentence-transformers"
            )
            return None
        log.info("Loading sentence-transformer %s (~118 MB)...", EMBED_MODEL_NAME)
        self._model = SentenceTransformer(EMBED_MODEL_NAME)
        log.info("Sentence-transformer loaded.")
        return self._model

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def ready(self) -> bool:
        return self._ready

    def retrieve(
        self,
        query: str,
        k: int = 5,
        lang: str = "en",
        min_score: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """Return up to k chunks most semantically similar to `query`.

        Each result is:
            {
                "id":       chunk id,
                "topic":    chunk topic,
                "text":     the retrieved text (in `lang` if available, else EN),
                "citation": formal citation string,
                "score":    cosine similarity, 0..1,
            }

        Results are sorted by score descending and filtered to score >= min_score.
        Returns [] if the index isn't loaded or the model isn't installed
        (graceful degradation — caller falls back to answer_bank).
        """
        if not self._ready or not query.strip():
            return []
        model = self._load_model()
        if model is None:
            return []

        q_vec = model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine == dot on unit vectors
        )[0].astype(np.float32)
        # cosine similarity via dot product (both sides normalised)
        scores = self._embeddings @ q_vec                       # [n]
        top_idx = np.argsort(-scores)[: k * 2]                  # over-pull to allow filter

        results: List[Dict[str, Any]] = []
        for i in top_idx:
            score = float(scores[i])
            if score < min_score:
                break
            chunk = self._chunks[int(i)]
            text_field = f"text_{lang}" if f"text_{lang}" in chunk else "text_en"
            results.append({
                "id":       chunk.get("id", ""),
                "topic":    chunk.get("topic", ""),
                "text":     chunk.get(text_field, chunk.get("text_en", "")),
                "citation": chunk.get("citation", ""),
                "score":    score,
            })
            if len(results) >= k:
                break
        return results


# --- module-level singleton -----------------------------------------------
_retriever: Optional[_Retriever] = None


def get_retriever() -> _Retriever:
    global _retriever
    if _retriever is None:
        _retriever = _Retriever()
    return _retriever


def retrieve(query: str, k: int = 5, lang: str = "en", **kw) -> List[Dict[str, Any]]:
    """Public helper — equivalent to get_retriever().retrieve()."""
    return get_retriever().retrieve(query, k=k, lang=lang, **kw)


def ready() -> bool:
    """True if the index loaded successfully at import time."""
    return get_retriever().ready()
