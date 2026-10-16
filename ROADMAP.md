# Roadmap

The goal of this project is a small but *production-shaped* RAG service — the
kind of retrieval quality, safety, and observability work that separates a real
system from a weekend demo. Items are scoped so each one is a focused change.

## Shipped

- [x] Token-aware chunking with overlap (tiktoken)
- [x] pgvector storage with an HNSW ANN index
- [x] Hybrid retrieval: vector + Postgres full-text, fused with Reciprocal Rank Fusion
- [x] LLM re-ranking of fused candidates
- [x] Grounded answers with inline citations and per-source relevance scores
- [x] Provider-agnostic (OpenAI / Gemini / Ollama via one OpenAI-compatible client)
- [x] Per-IP rate limiting on the LLM-backed endpoints
- [x] Two-pane chat UI (knowledge base + chat) deployed serverless on Vercel
- [x] Relevance gate — decline off-topic questions instead of hallucinating
- [x] Markdown rendering of answers in the UI

## Next (retrieval quality)

- [ ] Retrieval evaluation harness: labelled Q→source set, report hit-rate@k and MRR
- [ ] Maximal Marginal Relevance (MMR) to reduce near-duplicate chunks in context
- [ ] Near-duplicate chunk de-duplication at ingest time
- [ ] Contextual Retrieval: prepend an LLM-written chunk summary before embedding
- [ ] HyDE: embed a hypothetical answer to improve recall on sparse queries
- [ ] Sentence-aware chunk boundaries (avoid splitting mid-sentence)

## Next (product & UX)

- [ ] Stream answers token-by-token in the UI (the `/query/stream` endpoint exists)
- [ ] Multi-turn memory: contextualize follow-up questions against the conversation
- [ ] Click a citation to expand the full source chunk
- [ ] Thumbs up/down feedback on answers, stored for evaluation

## Next (operability)

- [ ] `/stats` endpoint: document & chunk counts, active provider/model
- [ ] Return latency and token usage in the query response metadata
- [ ] Deep health check that verifies the DB and embedding provider are reachable
- [ ] Optional semantic cache for repeated/similar queries
