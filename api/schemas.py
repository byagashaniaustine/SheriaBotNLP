"""Pydantic request/response models for the FastAPI service."""
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., example="Nimefutwa kazi bila notisi, nifanye nini?")


class ClassifyResponse(BaseModel):
    input: str
    intent: str
    confidence: Optional[float] = None
    top_3: List[Tuple[str, float]] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    dataset_rows: int
    intents_known: int
    graphs_available: int


class StatsResponse(BaseModel):
    total_rows: int
    unique_intents: int
    intent_counts: Dict[str, int]
    language_counts: Dict[str, int]
    avg_length_chars: float


class POSResponse(BaseModel):
    sample_size: int
    top_tags: Dict[str, int]
    tag_meanings: Dict[str, str]


class NERResponse(BaseModel):
    sample_size: int
    entity_counts: Dict[str, int]
    entity_examples: Dict[str, str]


class TopWordsResponse(BaseModel):
    top_words: List[Tuple[str, int]]
    total_tokens: int


class TfIdfResponse(BaseModel):
    top_terms: List[Tuple[str, float]]
    vocab_size_bow: int
    vocab_size_tfidf: int


class EmbeddingResponse(BaseModel):
    word: str
    vector_preview: List[float]  # first 10 dims
    vector_size: int
    similar_words: List[Tuple[str, float]]


class TokenisationResponse(BaseModel):
    input: str
    nltk_tokens: List[str]
    bert_subwords: List[str]
    nltk_count: int
    bert_count: int
    ratio: float


class TrainResponse(BaseModel):
    best_model: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    results: List[Dict[str, Any]]
