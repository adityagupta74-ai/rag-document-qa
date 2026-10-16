"""Application settings, loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- OpenAI (or any OpenAI-compatible provider: Gemini, Ollama, Groq...) ---
    openai_api_key: str
    openai_base_url: str | None = None   # set to use a non-OpenAI compatible endpoint
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # --- Database (Postgres + pgvector) ---
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    # --- RAG tuning ---
    chunk_size: int = 800        # tokens per chunk
    chunk_overlap: int = 120     # token overlap between chunks
    top_k: int = 4               # chunks passed to the LLM per query
    embedding_dim: int = 1536    # 1536 for text-embedding-3-small

    # --- Retrieval pipeline ---
    retrieval_candidates: int = 10  # candidates pulled per retriever before fusion/rerank
    use_hybrid: bool = True         # combine vector + keyword search via RRF
    use_rerank: bool = True         # LLM re-ranking of the fused candidates
    min_relevance: float = 0.12     # if the top match's cosine sim is below this,
                                    # decline instead of answering off-topic questions

    # --- Server / deploy ---
    run_db_init: bool = True        # bootstrap schema on startup; set false on serverless

    # --- Rate limiting (protects the public demo) ---
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 10  # per-IP requests/minute to /query and /documents

    @field_validator("database_url")
    @classmethod
    def _use_psycopg_driver(cls, value: str) -> str:
        """Accept plain Postgres URLs (Neon, Supabase, Heroku, etc.) and route
        them through the psycopg3 driver SQLAlchemy expects."""
        for prefix in ("postgresql://", "postgres://"):
            if value.startswith(prefix):
                return "postgresql+psycopg://" + value[len(prefix):]
        return value


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
