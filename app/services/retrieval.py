"""Retrieval: vector search, keyword (full-text) search, and hybrid fusion.

Vector search captures semantic similarity; keyword search captures exact terms
(names, IDs, acronyms) that embeddings often miss. Reciprocal Rank Fusion (RRF)
combines both rankings without needing the two score scales to be comparable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models import Chunk, Document
from app.services.embeddings import embed_query


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    content: str
    score: float  # meaning depends on stage: cosine sim, RRF score, or rerank score


def vector_search(db: Session, question: str, limit: int) -> list[RetrievedChunk]:
    """Semantic search via pgvector cosine distance."""
    query_embedding = embed_query(question)
    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    rows = db.execute(
        select(Chunk, Document, distance)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(limit)
    ).all()
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=document.id,
            document_title=document.title,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=round(1.0 - float(dist), 4),
        )
        for chunk, document, dist in rows
    ]


_WORD = re.compile(r"\w+")


def build_keyword_query(question: str) -> str:
    """Turn a natural-language question into an OR-style full-text query.

    ``websearch_to_tsquery`` ANDs every term, so a whole question almost never matches a
    single chunk. OR-ing the words lets any of them match, and ``ts_rank`` then orders
    chunks by how many (and how rare) of the terms they contain.
    """
    words = [w for w in _WORD.findall(question) if w.lower() != "or"]
    return " or ".join(words)


def keyword_search(db: Session, question: str, limit: int) -> list[RetrievedChunk]:
    """Lexical search via Postgres full-text search (ts_rank over a GIN index)."""
    query = build_keyword_query(question)
    if not query:
        return []
    sql = text(
        """
        SELECT c.id AS chunk_id, c.document_id, c.chunk_index, c.content,
               d.title AS document_title,
               ts_rank(to_tsvector('english', c.content),
                       websearch_to_tsquery('english', :q)) AS rank
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE to_tsvector('english', c.content) @@ websearch_to_tsquery('english', :q)
        ORDER BY rank DESC
        LIMIT :k
        """
    )
    rows = db.execute(sql, {"q": query, "k": limit}).mappings().all()
    return [
        RetrievedChunk(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            document_title=r["document_title"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            score=round(float(r["rank"]), 4),
        )
        for r in rows
    ]


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Combine multiple ranked lists into one via Reciprocal Rank Fusion.

    Each item's fused score is the sum of 1 / (k + rank) across the lists it
    appears in, so items ranked highly by *either* retriever rise to the top.
    """
    scores: dict[int, float] = {}
    by_id: dict[int, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id[item.chunk_id] = item

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        RetrievedChunk(
            chunk_id=(c := by_id[cid]).chunk_id,
            document_id=c.document_id,
            document_title=c.document_title,
            chunk_index=c.chunk_index,
            content=c.content,
            score=round(s, 5),
        )
        for cid, s in fused
    ]


def hybrid_search(db: Session, question: str, limit: int) -> list[RetrievedChunk]:
    """Vector + keyword search, merged with Reciprocal Rank Fusion."""
    vector = vector_search(db, question, limit)
    keyword = keyword_search(db, question, limit)
    return reciprocal_rank_fusion([vector, keyword])[:limit]
