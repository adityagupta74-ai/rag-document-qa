"""LLM-based re-ranking of retrieved candidates.

Retrieval is fast but coarse. A re-ranker reads each candidate against the
question and scores true relevance — catching passages that matched on vectors
or keywords but don't actually answer the question. Provider-agnostic: uses the
same chat model as generation, so it works on OpenAI, Gemini, or Ollama.
"""
from __future__ import annotations

import json
import logging

from app.config import get_settings
from app.services.openai_client import get_client
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM = "You are a precise search-relevance judge. Respond with JSON only."

# An 800-token chunk is ~3,500 characters; cutting passages shorter can hide the answer
# from the judge, which then scores everything 0.
PASSAGE_CHARS = 4000

# Calls where re-ranking fell back to retrieval order. The evaluation script reads this
# so it never reports numbers for a re-ranker that did not actually run.
stats = {"fallbacks": 0}


def _parse_scores(content: str, n: int) -> list[float] | None:
    """Parse a JSON array of n relevance scores (one per passage, in order).

    Tolerant of bare numbers ([8, 2, 5]) and objects ([{"score": 8}, ...]).
    """
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(content[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list):
        return None

    scores: list[float] = []
    for x in data:
        if isinstance(x, dict):
            x = x.get("score", x.get("relevance", 0))
        try:
            scores.append(float(x))
        except (TypeError, ValueError):
            scores.append(0.0)
    scores += [0.0] * (n - len(scores))
    return scores[:n]


def rerank(
    question: str, candidates: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """Re-score candidates by true relevance and return the top_k.

    Falls back to the original order if the model's response can't be parsed, so
    a flaky rerank never breaks the request.
    """
    if not candidates:
        return []

    listing = "\n\n".join(
        f'[{i}] (from "{c.document_title}") {c.content[:PASSAGE_CHARS]}'
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Passages:\n{listing}\n\n"
        f"Rate how well each passage answers the question, from 0 (irrelevant) to "
        f"10 (directly answers it). Return ONLY a JSON array of {len(candidates)} "
        f"numbers, in the same order as the passages — e.g. [8, 2, 5, 0]."
    )

    try:
        response = get_client().chat.completions.create(
            model=get_settings().openai_chat_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        scores = _parse_scores(response.choices[0].message.content or "", len(candidates))
        if scores is None:
            logger.warning("Rerank response could not be parsed; using retrieval order")
    except Exception as exc:
        logger.warning(
            "Rerank call failed (%s: %s); using retrieval order",
            type(exc).__name__,
            exc,
        )
        scores = None

    # If re-ranking is unavailable, keep retrieval order and scores (never all-zero).
    if scores and not any(s > 0 for s in scores):
        logger.warning("Rerank scored every passage 0; using retrieval order")
    if not scores or not any(s > 0 for s in scores):
        stats["fallbacks"] += 1
        return candidates[:top_k]

    ranked = sorted(
        zip(candidates, scores, strict=False), key=lambda cs: cs[1], reverse=True
    )
    return [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_title=c.document_title,
            chunk_index=c.chunk_index,
            content=c.content,
            score=round(float(s), 2),
        )
        for c, s in ranked[:top_k]
    ]
