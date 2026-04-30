from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceResponse(BaseModel):
    title: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    retrieved_chunks: int


class IndexResponse(BaseModel):
    article_count: int
    chunk_count: int


class HealthResponse(BaseModel):
    status: str

