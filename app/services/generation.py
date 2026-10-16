"""Answer generation grounded in retrieved context (with citations)."""
from __future__ import annotations

from collections.abc import Iterator

from app.config import get_settings
from app.services.openai_client import get_client
from app.services.retrieval import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided context. If the context does not contain the answer, say you "
    "don't know rather than guessing. Cite sources inline as [n], where n is "
    "the context number."
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f'[{i}] (from "{c.document_title}")\n{c.content}'
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks) or "No context available."


def _messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{_build_context(chunks)}\n\nQuestion: {question}",
        },
    ]


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    response = get_client().chat.completions.create(
        model=get_settings().openai_chat_model,
        messages=_messages(question, chunks),
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def stream_answer(question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
    stream = get_client().chat.completions.create(
        model=get_settings().openai_chat_model,
        messages=_messages(question, chunks),
        temperature=0.2,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta
