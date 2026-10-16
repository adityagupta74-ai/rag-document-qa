"""OpenAI embedding helpers (batched)."""
from __future__ import annotations

from app.config import get_settings
from app.services.openai_client import get_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in a single API call. Order is preserved.

    Requests ``embedding_dim`` dimensions so models that support Matryoshka
    truncation (OpenAI text-embedding-3-*, Gemini gemini-embedding-001) return
    vectors that match the database column.
    """
    if not texts:
        return []
    settings = get_settings()
    response = get_client().embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
        dimensions=settings.embedding_dim,
    )
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
