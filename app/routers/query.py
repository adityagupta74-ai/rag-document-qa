"""RAG query endpoints (JSON answer + streaming)."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.schemas import Citation, QueryRequest, QueryResponse
from app.services.generation import generate_answer, stream_answer
from app.services.retrieval import RetrievedChunk
from app.services.search import search

router = APIRouter(prefix="/query", tags=["query"])

# Returned (instead of an LLM-generated guess) when retrieval confidence is low.
NO_ANSWER = (
    "I couldn't find anything about that in the current knowledge base. "
    "Try rephrasing, or add a document that covers it."
)

# The streaming response is "answer text" + this byte + "citations JSON". 0x1F
# (ASCII unit separator) never appears in natural-language output, so the client
# can split on it without escaping.
CITATION_SEPARATOR = "\x1f"


def _citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            document_id=c.document_id,
            document_title=c.document_title,
            chunk_index=c.chunk_index,
            score=c.score,
            snippet=c.content[:240].strip(),
        )
        for c in chunks
    ]


@router.post("", response_model=QueryResponse)
def query(payload: QueryRequest, db: Session = Depends(get_db)):
    """Retrieve relevant context and answer with inline citations."""
    settings = get_settings()
    top_k = payload.top_k or settings.top_k
    result = search(db, payload.question, top_k)
    if result.confidence < settings.min_relevance:
        return QueryResponse(answer=NO_ANSWER, citations=[])
    answer = generate_answer(payload.question, result.chunks)
    return QueryResponse(answer=answer, citations=_citations(result.chunks))


@router.post("/stream")
def query_stream(payload: QueryRequest, db: Session = Depends(get_db)):
    """Stream the answer token-by-token, then a trailing JSON citations frame.

    Wire format: the answer text, then CITATION_SEPARATOR, then a JSON object
    `{"citations": [...]}`. Streaming the text first keeps the UI responsive;
    citations arrive once, at the end, when the source chunks are known.
    """
    settings = get_settings()
    top_k = payload.top_k or settings.top_k
    result = search(db, payload.question, top_k)

    def frame(citations: list[Citation]) -> str:
        data = {"citations": [c.model_dump() for c in citations]}
        return CITATION_SEPARATOR + json.dumps(data)

    if result.confidence < settings.min_relevance:
        def gated():
            yield NO_ANSWER
            yield frame([])
        return StreamingResponse(gated(), media_type="text/plain; charset=utf-8")

    citations = _citations(result.chunks)

    def generate():
        yield from stream_answer(payload.question, result.chunks)
        yield frame(citations)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
