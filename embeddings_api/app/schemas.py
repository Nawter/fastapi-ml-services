from pydantic import BaseModel, Field, computed_field
from typing import List, Optional


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)


class EmbedBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64)


class EmbedResult(BaseModel):
    text: str
    embedding: List[float]
    dim: int


class SimilarityRequest(BaseModel):
    text_a: str = Field(..., min_length=1, max_length=512)
    text_b: str = Field(..., min_length=1, max_length=512)


class SimilarityResult(BaseModel):
    text_a: str
    text_b: str
    similarity: float  # 0.0 to 1.0

    @computed_field
    @property
    def interpretation(self) -> str:
        if self.similarity >= 0.9:
            return "Very similar"
        if self.similarity >= 0.7:
            return "Similar"
        if self.similarity >= 0.5:
            return "Somewhat related"
        return "Not similar"
