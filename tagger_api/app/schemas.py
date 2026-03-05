from pydantic import BaseModel, Field
from typing import List, Optional


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=2048)


class BatchProcessRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=16)


class EntityResult(BaseModel):
    text: str   # the entity string e.g. "Apple"
    type: str   # ORG, PER, LOC, MISC
    score: float


class ProcessResult(BaseModel):
    original_text: str
    summary: str
    sentiment: str
    sentiment_score: float
    entities: List[EntityResult]
