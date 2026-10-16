"""Document ingestion endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Chunk, Document
from app.schemas import DocumentSummary, IngestRequest, IngestResponse
from app.services.extract import extract_text
from app.services.ingestion import ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentSummary])
def list_documents(db: Session = Depends(get_db)):
    """List ingested documents and their chunk counts (powers the sidebar)."""
    rows = db.execute(
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    ).all()
    return [
        DocumentSummary(id=d.id, title=d.title, source=d.source, chunks=n)
        for d, n in rows
    ]


def _to_response(document) -> IngestResponse:
    return IngestResponse(
        document_id=document.id,
        title=document.title,
        chunks_created=len(document.chunks),
    )


@router.post("", response_model=IngestResponse, status_code=201)
def create_document(payload: IngestRequest, db: Session = Depends(get_db)):
    """Ingest a raw-text document."""
    try:
        document = ingest_document(db, payload.title, payload.content, payload.source)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _to_response(document)


@router.post("/upload", response_model=IngestResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    """Ingest an uploaded file (.txt, .md, .pdf)."""
    content = await extract_text(file)
    try:
        document = ingest_document(
            db, file.filename or "upload", content, file.filename
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _to_response(document)
