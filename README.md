# 🔎 RAG Document Q&A Service — FastAPI + pgvector

> Based on [rag-starter-fastapi](https://github.com/dhairyasenjaliya/rag-starter-fastapi) by Dhairya Senjaliya (MIT). See [What I changed](#what-i-changed).

A small, **production-shaped** Retrieval-Augmented Generation (RAG) service.
Ingest documents, embed and store them in **Postgres/pgvector**, then ask
questions and get **grounded, cited answers** — with streaming support.

Built to be read: typed, layered, tested, and runnable in two commands.

<p>
  <a href="https://github.com/adityagupta74-ai/rag-document-qa/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adityagupta74-ai/rag-document-qa/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white">
  <img alt="pgvector" src="https://img.shields.io/badge/pgvector-HNSW-4169E1?logo=postgresql&logoColor=white">
  <img alt="Providers" src="https://img.shields.io/badge/LLM-OpenAI%20%C2%B7%20Gemini%20%C2%B7%20Ollama-412991">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

> 🆓 **Runs free:** works with OpenAI, or **Google Gemini's free tier** / local **Ollama** — see [Providers](#providers).

---

## Why this exists

Most "RAG demos" are a single notebook that falls apart the moment you need an
API, a real database, or citations you can trust. This is the opposite: a clean,
minimal **backend you could actually build on** — clear separation between
chunking, embedding, retrieval, and generation, with a real vector store and an
ANN index.

---

## How it works

```
                 ┌─────────────┐   chunk + embed    ┌──────────────────────┐
   Document ───▶ │  Ingestion  │ ─────────────────▶ │  Postgres + pgvector │
                 └─────────────┘                    │   (HNSW cosine ANN)  │
                                                     └──────────┬───────────┘
                                                                │ top-k
   Question ───▶ embed ───▶ vector search ──────────────────────┘
                                                  │
                                                  ▼
                                   ┌────────────────────────────┐
                                   │  Generation (grounded LLM) │ ─▶ answer + citations
                                   └────────────────────────────┘
```

1. **Ingest** — text is split into token-bounded, overlapping chunks (`tiktoken`).
2. **Embed** — each chunk is embedded via your provider (OpenAI, Gemini, or Ollama).
3. **Store** — chunks + vectors land in Postgres with a pgvector **HNSW** index.
4. **Retrieve & re-rank** — hybrid search (vector + keyword, fused with RRF),
   then an LLM re-ranks the candidates. See [Retrieval pipeline](#retrieval-pipeline).
5. **Generate** — retrieved context is fed to the chat model, which answers
   **only from context** and cites sources as `[n]`.

---

## Retrieval pipeline

![Hybrid retrieval pipeline](.github/rag-pipeline.png)

Naive RAG runs a single vector search and hopes for the best. This uses a
two-stage pipeline that's far more robust:

1. **Hybrid retrieval** — **vector search** (pgvector, semantic) and **keyword
   search** (Postgres full-text, exact terms) run in parallel, then merge via
   **Reciprocal Rank Fusion**. Embeddings miss exact names, IDs, and acronyms;
   keyword search catches them — together they beat either alone.
2. **LLM re-ranking** — the fused candidates are scored for true relevance by the
   model, filtering out passages that matched superficially before they reach the
   answer.

Both stages are toggleable in `.env`:

```
USE_HYBRID=true            # vector + keyword (RRF)
USE_RERANK=true            # LLM re-ranking
RETRIEVAL_CANDIDATES=10    # candidates per retriever before fusion/rerank
```

A separate **relevance gate** (`MIN_RELEVANCE`) protects answer quality: if the
best retrieved chunk's cosine similarity is too low, the API returns "not in the
knowledge base" instead of letting the model answer from irrelevant context —
and skips the LLM call entirely.

---

## Evaluating retrieval quality

Retrieval claims shouldn't be taken on faith — they should be measured. The repo
ships a small evaluation harness over a labelled question set (across the bundled
handbook, FAQ, and onboarding docs). Each question is labelled with a distinctive
answer phrase, so it scores the rank of the **exact passage that answers it** —
not just the right document — and compares the full pipeline against a vector-only
baseline. Several questions are *lexical traps* (AES-256, SEPA, `make bootstrap`,
429) — exact tokens embeddings tend to blur but keyword search nails.

```bash
make eval        # python -m scripts.eval  (examples auto-ingested if the store is empty)
```

```text
=== Vector-only baseline (top_k=5) ===
  → hit-rate@5: 16/16 = 1.00    MRR: 0.969

=== Hybrid + re-ranking (top_k=5) ===
  → hit-rate@5: 16/16 = 1.00    MRR: 1.000
```

> A representative run on the bundled corpus with Google Gemini embeddings. Both
> pipelines find every answer in the top 5, but **re-ranking lifts MRR from 0.969
> to a perfect 1.000** — it promotes the precise answer passage to rank 1 in the
> case where plain vector search left it at rank 2. Numbers vary by provider and
> run; the gap widens on larger, denser corpora. The real win is the workflow: a
> labelled set + metrics, so changes to chunking, fusion, or re-ranking are judged
> objectively rather than by vibes.

---

## Quickstart

Requires Python 3.12, a Postgres database with **pgvector**, and an API key
(OpenAI, or **free** Gemini / Ollama — see [Providers](#providers)).

```bash
# 1. Configure — set your API key + DATABASE_URL
cp .env.example .env

# 2. Get a pgvector database — either:
make db                     # (a) local Postgres+pgvector via Docker, OR
#                             (b) a hosted DB — set DATABASE_URL in .env to a
#                                 Neon / Supabase URL (both ship pgvector)

# 3. Install deps & run
make install
make run                    # http://localhost:8000/docs

# 4. Load sample docs and ask a question
make examples               # ingest the bundled example docs
make seed                   # or: ingest one doc + run a query end-to-end
```

Open **http://localhost:8000** for the built-in **chat UI**, or
**/docs** for the interactive Swagger API explorer.

---

## Built-in chat demo

The app ships a zero-build chat UI (served at `/`) so you can try RAG in the
browser — ask a question and watch the answer **stream in token-by-token**, then
get **sources and relevance scores**. Answers render Markdown, and the streaming
endpoint sends citations in a trailing frame so they appear the moment generation
finishes:

![Chat UI](.github/rag-ui.png)

---

## Deploy to Vercel (free)

The app runs on Vercel's free tier as a **Python serverless function** — a fully
dynamic, public demo at zero cost.

1. Import this repo at **vercel.com/new**.
2. Add these **Environment Variables** in the project settings:

   | Variable | Value |
   |----------|-------|
   | `OPENAI_API_KEY` | your Gemini (or OpenAI) key |
   | `OPENAI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
   | `OPENAI_EMBEDDING_MODEL` | `gemini-embedding-001` |
   | `OPENAI_CHAT_MODEL` | `gemini-3.8-flash` |
   | `EMBEDDING_DIM` | `768` |
   | `DATABASE_URL` | your Neon connection string |
   | `RUN_DB_INIT` | `false` — schema is already provisioned |

3. Deploy → live at `https://<project>.vercel.app`.

Seed your database once locally (`make examples`) so the demo has content.
`vercel.json` routes all traffic to the FastAPI app and allows 60s/request
(hybrid retrieval + re-rank + generation makes a few model calls).

> 🛡️ The LLM-backed endpoints (`/query`, `/documents`) are protected by a
> **per-IP rate limit** (DB-backed, so it works across serverless instances).
> Tune with `RATE_LIMIT_PER_MINUTE` or turn it off with `RATE_LIMIT_ENABLED=false`.

---

## Providers

The app talks to any **OpenAI-compatible** API via `OPENAI_BASE_URL`, so you can
run it on a paid OpenAI key — or **for free**:

| Provider | Cost | Key `.env` settings |
|----------|------|---------------------|
| **OpenAI** | paid | `OPENAI_EMBEDDING_MODEL=text-embedding-3-small` · `EMBEDDING_DIM=1536` |
| **Google Gemini** | **free tier** | `OPENAI_BASE_URL=…/v1beta/openai/` · `OPENAI_EMBEDDING_MODEL=gemini-embedding-001` · `OPENAI_CHAT_MODEL=gemini-3.8-flash` · `EMBEDDING_DIM=768` |
| **Ollama** | **free, local** | `OPENAI_BASE_URL=http://localhost:11434/v1` · `nomic-embed-text` / `llama3.2` · `EMBEDDING_DIM=768` |

Copy-paste blocks for each are in [`.env.example`](.env.example). Embedding
models that support Matryoshka truncation (OpenAI `text-embedding-3-*`, Gemini
`gemini-embedding-001`) return vectors sized to `EMBEDDING_DIM`.

---

## Example documents

The repo ships small, realistic sample docs in [`examples/`](examples/) so you
can see real multi-chunk retrieval immediately:

| File | Format | Good questions to ask |
|------|--------|-----------------------|
| `company-handbook.md` | Markdown | "How many PTO days do I get?" · "What's the laptop security policy?" |
| `product-faq.txt` | Plain text | "What are the API rate limits?" · "Can I export my data?" |
| `onboarding-guide.pdf` | PDF | "When do engineers join on-call?" · "What's the git workflow?" |

```bash
make examples        # ingest all three bundled docs
```

### Testing with large documents

Large files are **not** committed to the repo (good git hygiene). Instead, a
script downloads a full public-domain book from Project Gutenberg into the
git-ignored `examples/_large/` folder to demonstrate large-document ingestion:

```bash
make large                                   # ~3.3 MB book, downloaded not committed
python -m scripts.fetch_large_sample --ingest  # download AND embed the whole book
```

---

## API

### Ingest a document
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
        "title": "Refund Policy",
        "content": "Customers may request a full refund within 30 days...",
        "source": "policy.txt"
      }'
```
```json
{ "document_id": 1, "title": "Refund Policy", "chunks_created": 1 }
```

### Upload a file (.txt, .md, .pdf)
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@handbook.pdf"
```

### Ask a question (grounded + cited)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{ "question": "How long do I have to get a refund?" }'
```
```json
{
  "answer": "You have 30 days from purchase to request a full refund [1].",
  "citations": [
    {
      "document_id": 1,
      "document_title": "Refund Policy",
      "chunk_index": 0,
      "score": 0.83,
      "snippet": "Customers may request a full refund within 30 days..."
    }
  ]
}
```

### Stream the answer
```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{ "question": "How long do I have to get a refund?" }'
```

---

## Project structure

```
app/
├── main.py                 # FastAPI app + lifespan (DB bootstrap)
├── config.py               # Settings (pydantic-settings)
├── database.py             # Engine, session, pgvector + HNSW init
├── models.py               # Document / Chunk ORM models
├── schemas.py              # Request/response models
├── routers/
│   ├── documents.py        # ingest + upload
│   └── query.py            # query + streaming
└── services/
    ├── chunking.py         # token-aware splitting w/ overlap
    ├── embeddings.py       # batched embeddings (dimension-aware)
    ├── ingestion.py        # chunk → embed → persist
    ├── retrieval.py        # vector + keyword search, RRF fusion
    ├── reranking.py        # LLM re-ranking of candidates
    ├── search.py           # retrieve → rerank orchestration
    ├── generation.py       # grounded answer + citations
    ├── extract.py          # txt / md / pdf text extraction
    ├── openai_client.py    # shared, provider-agnostic client
    └── static/index.html   # built-in chat demo UI
api/index.py                # Vercel serverless entrypoint
vercel.json                 # Vercel routing + function config
scripts/                    # seed · ingest_examples · fetch_large_sample
tests/                      # offline unit + smoke tests
```

---

## Design decisions

- **pgvector over a bolt-on vector DB** — one database for app data *and*
  vectors means simpler ops, real SQL joins between chunks and their documents,
  and transactional ingestion.
- **HNSW index** (`vector_cosine_ops`) — fast approximate nearest-neighbour
  search that stays quick as the corpus grows.
- **Hybrid retrieval + re-ranking** — vector and keyword search merged with RRF,
  then LLM re-ranking. Catches the exact-term matches embeddings miss, and the
  relevance failures retrieval alone can't.
- **Token-aware chunking with overlap** — chunks respect the embedding model's
  context and don't sever ideas mid-sentence.
- **Strictly grounded generation** — the model is instructed to answer only
  from retrieved context and to say "I don't know" rather than hallucinate.
- **Batched embeddings** — one API call per document, not one per chunk.

---

## Testing & CI

```bash
make lint        # ruff check .
make test        # pytest -q
```

Tests (chunking, RRF fusion, the relevance gate, health) run **fully offline** —
no API key or database needed — by stubbing the DB session, rate limiter, and
LLM. Every push and PR runs **ruff + pytest** via
[GitHub Actions](.github/workflows/ci.yml), so the badge up top reflects the real
state of `main`.

---

## Extending it

- Switch providers/models freely in `.env` — just keep `EMBEDDING_DIM` in sync
  (note: pgvector's HNSW index supports ≤2000 dims; above that the app keeps the
  column but skips the index automatically).
- Add metadata filtering (tenant, source, date) to the retrieval query.
- Add a re-ranker between retrieval and generation.
- Put it behind auth + rate limiting for multi-tenant use.

---

## What I changed

This project started from the MIT-licensed
[rag-starter-fastapi](https://github.com/dhairyasenjaliya/rag-starter-fastapi) by
Dhairya Senjaliya. My changes on top of it:

- **Reranker error handling** — failures were swallowed silently (a retired model name went unnoticed). Failures, unparseable replies and all-zero scores are now logged.
- **Reranker input fix** — passages were truncated to 600 characters, hiding answers deeper in 800-token chunks and making the judge score everything 0. It now sees up to 4,000 characters.
- **Keyword search fix** — full-text search required every word of the question to match, so natural-language questions returned nothing and "hybrid" retrieval was effectively vector-only. Query terms are now OR-ed.
- **Heading-aware Markdown chunking** — chunks start at section boundaries; oversized sections are split with their heading repeated on each piece.
- **`DELETE /documents/{id}`** endpoint with offline tests.
- **Evaluation harness** — added paraphrased questions, a hybrid-without-rerank variant, an embedding cache and throttling for free-tier rate limits.
- **Tests** — 15 new offline tests (29 total), all run in CI.

### Evaluation

24 labelled questions over the 3 sample documents (about 30 chunks at 120 tokens), top-3:

| Pipeline | hit-rate@3 | MRR |
|---|---|---|
| Vector-only | 0.96 (23/24) | 0.896 |
| Hybrid (vector + keyword, RRF) | 1.00 (24/24) | 0.951 |

This is a small, synthetic test set, so treat the numbers as a regression check rather than a benchmark. LLM re-ranking has not been measured yet (the free-tier quota ran out during the run).

Reproduce:

```bash
docker compose exec db createdb -U rag rag_eval
DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/rag_eval \
CHUNK_SIZE=120 CHUNK_OVERLAP=20 EVAL_TOP_K=3 python -m scripts.eval
```

---

## About

Maintained by **Aditya Gupta** — [GitHub](https://github.com/adityagupta74-ai).

## License

MIT — see [LICENSE](LICENSE). Original work © 2026 Dhairya Senjaliya; modifications © 2026 Aditya Gupta.
