"""Pydantic request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    title: str = Field(..., examples=["Company Handbook"])
    content: str = Field(..., min_length=1)
    source: str | None = Field(None, examples=["handbook.pdf"])


class IngestResponse(BaseModel):
    document_id: int
    title: str
    chunks_created: int


class DocumentSummary(BaseModel):
    id: int
    title: str
    source: str | None
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the refund policy?"])
    top_k: int | None = Field(None, ge=1, le=20)


class Citation(BaseModel):
    document_id: int
    document_title: str
    chunk_index: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
