"""Database engine, session, and schema/extension bootstrap."""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import get_settings

settings = get_settings()

# prepare_threshold=None disables psycopg3 prepared statements so the app works
# over pgbouncer/pooled endpoints (Neon, Supabase) as well as direct connections.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
    connect_args={"prepare_threshold": None},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db() -> None:
    """Enable pgvector, create tables, and add an ANN index.

    Idempotent — safe to call on every startup.
    """
    # Import models so they register on Base.metadata before create_all.
    from app import models  # noqa: F401

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)

    # HNSW index for fast approximate nearest-neighbour search at scale.
    # pgvector's HNSW index supports at most 2000 dimensions; above that we keep
    # the column (exact search) but skip the index.
    if settings.embedding_dim <= 2000:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                    "ON chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )

    # GIN index powering keyword (full-text) search for hybrid retrieval.
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_content_fts "
                "ON chunks USING gin (to_tsvector('english', content))"
            )
        )


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
