"""Extract plain text from uploaded files (.txt, .md, .pdf)."""
from __future__ import annotations

import io

from fastapi import UploadFile
from pypdf import PdfReader


async def extract_text(file: UploadFile) -> str:
    data = await file.read()
    name = (file.filename or "").lower()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return data.decode("utf-8", errors="ignore")
