from pydantic import BaseModel, Field, computed_field
from typing import List, Optional


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Input text to classify")
    top_k: int = Field(default=1, ge=1, le=2)


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32)


class ClassifyResult(BaseModel):
    label: str
    score: float
    inference_ms: Optional[float] = None

    @computed_field
    @property
    def confidence_pct(self) -> str:
        return f"{self.score * 100:.1f}%"

    @computed_field
    @property
    def is_high_confidence(self) -> bool:
        return self.score >= 0.85


class LabelScore(BaseModel):
    label: str
    score: float


class ClassifyResultTopK(BaseModel):
    text: str
    predictions: List[LabelScore]


class BatchClassifyResult(BaseModel):
    results: List[ClassifyResult]
    total: int
