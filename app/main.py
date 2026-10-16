"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app.config import get_settings
from app.database import init_db
from app.routers import documents, query
from app.services.ratelimit import check_rate_limit

STATIC_DIR = Path(__file__).parent / "static"
RATE_LIMITED_PREFIXES = ("/query", "/documents")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On serverless, the schema is already provisioned — skip DDL on cold start.
    # Never let a startup DB hiccup crash the whole function; log and continue.
    if get_settings().run_db_init:
        try:
            init_db()
        except Exception as exc:  # noqa: BLE001
            print(f"[startup] init_db skipped: {exc}")
    yield


app = FastAPI(
    title="RAG Starter — FastAPI + pgvector",
    description=(
        "A small, production-shaped Retrieval-Augmented Generation API: "
        "ingest documents, embed & store them in Postgres/pgvector, then "
        "answer questions with cited, grounded responses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(query.router)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Per-IP rate limit on the expensive (LLM-backed) POST endpoints."""
    if request.method == "POST" and request.url.path.startswith(RATE_LIMITED_PREFIXES):
        forwarded = request.headers.get("x-forwarded-for", "")
        ip = forwarded.split(",")[0].strip() or (
            request.client.host if request.client else "unknown"
        )
        try:
            allowed = check_rate_limit(ip)
        except Exception:  # noqa: BLE001 — never block the app on a limiter error
            allowed = True
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded — please wait a minute."},
            )
    return await call_next(request)


@app.get("/", include_in_schema=False)
def home():
    """Serve the interactive chat demo UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}
