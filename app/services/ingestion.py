"""Document ingestion: chunk -> embed -> persist."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Chunk, Document
from app.services.chunking import chunk_markdown, chunk_text
from app.services.embeddings import embed_texts


def ingest_document(
    db: Session, title: str, content: str, source: str | None
) -> Document:
    settings = get_settings()
    is_markdown = (source or "").lower().endswith((".md", ".markdown"))
    chunker = chunk_markdown if is_markdown else chunk_text
    pieces = chunker(content, settings.chunk_size, settings.chunk_overlap)
    if not pieces:
        raise ValueError("Document produced no chunks (empty content?)")

    embeddings = embed_texts(pieces)

    document = Document(title=title, source=source)
    document.chunks = [
        Chunk(chunk_index=i, content=piece, embedding=embedding)
        for i, (piece, embedding) in enumerate(zip(pieces, embeddings, strict=True))
    ]
    db.add(document)
    db.commit()
    db.refresh(document)
    return document
