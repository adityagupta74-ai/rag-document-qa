"""Retrieval evaluation harness.

Measures how often retrieval surfaces the *exact passage that answers* a set of
labelled questions, and compares the full hybrid + re-ranking pipeline against a
plain vector-search baseline. This is the difference between "it feels good" and
"the answer chunk ranks first 15/16 times, MRR 0.93".

Each case is labelled with a distinctive answer phrase; a question "hits" when a
retrieved chunk actually contains that phrase. Several cases are lexical traps —
exact tokens (AES-256, SEPA, `make bootstrap`, 429) that embeddings blur but
keyword search catches — so the hybrid pipeline can show its edge.

Metrics:
  * hit-rate@k — fraction of questions whose answer chunk appears in the top k
  * MRR        — mean reciprocal rank of the answer chunk (1/rank, 0 if absent)

Usage:
    python -m scripts.eval

Requires a running database (`make db`) and OPENAI_API_KEY in your env/.env.
The bundled example documents are ingested automatically if the store is empty.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import func, select

from app.config import get_settings
from app.database import SessionLocal, init_db
from app.models import Document
from app.services import reranking, retrieval
from app.services.ingestion import ingest_document
from app.services.search import search
from scripts.ingest_examples import EXAMPLES, SUPPORTED, read_file


@dataclass
class Case:
    question: str
    answer: str  # a distinctive phrase the answer-bearing chunk must contain


# Chunk-level labels: each question is scored on the rank of the *specific
# passage* that answers it (not just the right document). The second block are
# lexical traps — questions whose answer hinges on an exact token (AES-256, SEPA,
# `make bootstrap`, "429") that embeddings tend to blur but keyword search nails.
CASES: list[Case] = [
    Case("How many paid vacation days do full-time employees get?", "20 days of paid vacation"),
    Case("How much parental leave do new parents get?", "16 weeks of fully paid parental leave"),
    Case("What is the annual home-office equipment budget?", "$1,000 annual"),
    Case("How soon must a security incident be reported?", "within one hour of discovery"),
    Case("What are the core collaboration hours?", "10:00 AM to"),
    Case("What plans does Acme Cloud offer?", "three plans"),
    Case("Can I export my data, and in what format?", "JSON or CSV"),
    Case("When do engineers join the on-call rotation?", "after their first month"),
    Case("What git workflow does the engineering team use?", "trunk-based development"),
    Case("How quickly must you acknowledge a page when on-call?", "within 15 minutes"),
    # Lexical traps — exact tokens that reward keyword search + re-ranking.
    Case("What encryption standard protects data at rest?", "AES-256"),
    Case("What HTTP status code signals you exceeded the rate limit?", "429"),
    Case("Which payment networks are supported for bank transfer?", "SEPA"),
    Case("What command installs the local toolchain?", "make bootstrap"),
    Case("What is the free-plan API request limit per minute?", "60 requests per minute"),
    Case("How many days do customers have for a no-questions-asked refund?", "within 30 days"),
    # Paraphrased questions: little word overlap with the answer, so they need real
    # semantic matching and are harder than the ones above.
    Case(
        "Can I still get money back if I ask after the first month?",
        "store credit only",
    ),
    Case(
        "Who steps in when coworkers cannot resolve a dispute?",
        "available to mediate",
    ),
    Case(
        "From what spend amount must I keep a receipt?",
        "over $25",
    ),
    Case(
        "How fast do enterprises get a reply to a critical outage?",
        "first response within one hour",
    ),
    Case(
        "Do students have to pay for the Pro plan?",
        "free Pro access",
    ),
    Case(
        "How long does the service keep retrying a failed webhook?",
        "retried for up to 24 hours",
    ),
    Case(
        "Can I move a project from the US to the EU region later?",
        "cannot be changed after creation",
    ),
    Case(
        "Which TLS versions protect data in transit?",
        "TLS 1.2+",
    ),
]

TOP_K = int(os.environ.get("EVAL_TOP_K", "5"))
# Seconds to wait after each re-ranked query, so free-tier LLM quotas
# (e.g. 5 requests/minute) don't reject the re-rank call.
EVAL_DELAY = float(os.environ.get("EVAL_DELAY", "0"))


def _ensure_examples(db) -> None:
    """Ingest the bundled examples once, so the harness works from a clean store."""
    if db.execute(select(func.count(Document.id))).scalar_one() > 0:
        return
    print("Store is empty — ingesting bundled example documents...\n")
    for path in sorted(EXAMPLES.glob("*")):
        if path.name.startswith("_") or path.suffix.lower() not in SUPPORTED:
            continue
        ingest_document(db, path.stem, read_file(path), path.name)


def _answer_rank(db, case: Case, k: int) -> int:
    """1-based rank of the first retrieved chunk that contains the answer, or 0."""
    result = search(db, case.question, top_k=k)
    if get_settings().use_rerank and EVAL_DELAY:
        time.sleep(EVAL_DELAY)
    needle = case.answer.lower()
    for rank, chunk in enumerate(result.chunks, start=1):
        if needle in chunk.content.lower():
            return rank
    return 0


def _evaluate(db, label: str, k: int) -> tuple[float, float]:
    hits = 0
    rr_total = 0.0
    print(f"\n=== {label} (top_k={k}) ===")
    for case in CASES:
        rank = _answer_rank(db, case, k)
        hit = rank != 0
        hits += int(hit)
        rr_total += (1.0 / rank) if hit else 0.0
        mark = f"rank {rank}" if hit else "miss"
        print(f"  [{mark:>7}] {case.question}")
    n = len(CASES)
    hit_rate, mrr = hits / n, rr_total / n
    print(f"  → hit-rate@{k}: {hits}/{n} = {hit_rate:.2f}    MRR: {mrr:.3f}")
    return hit_rate, mrr


def main() -> None:
    init_db()
    # Each question is embedded once per pipeline variant; cache to save API calls.
    retrieval.embed_query = lru_cache(maxsize=None)(retrieval.embed_query)
    db = SessionLocal()
    try:
        _ensure_examples(db)
        settings = get_settings()

        # Baseline: pure vector (semantic) search, no fusion or re-ranking.
        settings.use_hybrid = False
        settings.use_rerank = False
        base = _evaluate(db, "Vector-only baseline", TOP_K)

        # Hybrid retrieval (vector + keyword via RRF), no re-ranking: isolates keyword search.
        settings.use_hybrid = True
        settings.use_rerank = False
        hybrid = _evaluate(db, "Hybrid (no re-rank)", TOP_K)

        # Full pipeline: hybrid retrieval (vector + keyword via RRF) + LLM re-rank.
        settings.use_hybrid = True
        settings.use_rerank = True
        reranking.stats["fallbacks"] = 0
        full = _evaluate(db, "Hybrid + re-ranking", TOP_K)
        fallbacks = reranking.stats["fallbacks"]

        print("\nSummary")
        print(f"  vector-only        hit-rate {base[0]:.2f}   MRR {base[1]:.3f}")
        print(f"  hybrid (no rerank) hit-rate {hybrid[0]:.2f}   MRR {hybrid[1]:.3f}")
        if fallbacks:
            print(
                "  hybrid + re-rank   NOT VALID: re-ranking fell back to retrieval order "
                f"on {fallbacks}/{len(CASES)} questions"
            )
            raise SystemExit(1)
        print(f"  hybrid + re-rank   hit-rate {full[0]:.2f}   MRR {full[1]:.3f}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
