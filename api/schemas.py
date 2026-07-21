"""Pydantic request/response models for the FastAPI service."""
from __future__ import annotations
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., examples=["Nimefutwa kazi bila notisi, nifanye nini?"])


class ClassifyResponse(BaseModel):
    input: str
    intent: str
    confidence: Optional[float] = None
    top_3: List[Tuple[str, float]] = []
    answer: str
    lang: str
    source: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str
    num_labels: int
    answer_bank_loaded: bool
