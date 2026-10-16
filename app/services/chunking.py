"""Token-aware text chunking with overlap.

Splitting on token boundaries keeps every chunk safely within the embedding
model's context window, while the overlap preserves meaning that would
otherwise be lost across a hard cut.
"""
from __future__ import annotations

import os
import re

# Serverless filesystems are read-only except /tmp — point tiktoken's cache there
# so the encoder can download/cache on a cold start (e.g. Vercel).
os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/tmp/tiktoken-cache")

import tiktoken  # noqa: E402

_encoder = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    tokens = _encoder.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        piece = _encoder.decode(window).strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(tokens):
            break
    return chunks


_HEADING = re.compile(r"^#{1,6}\s+\S")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def split_markdown_sections(text: str) -> list[str]:
    """Split Markdown into sections, each starting at a heading line.

    Lines inside fenced code blocks are never treated as headings, so a `# comment`
    in a code sample does not start a new section.
    """
    sections: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        if not in_fence and _HEADING.match(line) and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section]


def chunk_markdown(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Heading-aware chunking for Markdown.

    Whole sections are packed together up to ``chunk_size`` tokens. A section that is
    larger than ``chunk_size`` is split with ``chunk_text``, and its heading is repeated
    on every piece after the first so each chunk keeps its context.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append("\n\n".join(buffer))
        buffer, buffer_tokens = [], 0

    for section in split_markdown_sections(text):
        size = _count_tokens(section)
        if size > chunk_size:
            flush()
            heading = section.splitlines()[0] if _HEADING.match(section) else ""
            for i, piece in enumerate(chunk_text(section, chunk_size, chunk_overlap)):
                if i > 0 and heading and not piece.startswith(heading):
                    piece = f"{heading}\n{piece}"
                chunks.append(piece)
            continue
        if buffer and buffer_tokens + size > chunk_size:
            flush()
        buffer.append(section)
        buffer_tokens += size
    flush()
    return chunks
