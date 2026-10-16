"""Re-ranker tests — offline, with a stubbed LLM client."""
import logging
from types import SimpleNamespace

from app.services import reranking
from app.services.retrieval import RetrievedChunk


def _chunk(cid, content):
    return RetrievedChunk(
        chunk_id=cid,
        document_id=1,
        document_title="doc",
        chunk_index=cid,
        content=content,
        score=0.01,
    )


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_prompt_includes_text_past_the_old_600_char_cutoff(monkeypatch):
    late = "The answer is forty-two."
    client = FakeClient("[9]")
    monkeypatch.setattr(reranking, "get_client", lambda: client)
    reranking.rerank("q", [_chunk(1, "x " * 500 + late)], 1)
    assert late in client.prompts[0]


def test_candidates_are_ordered_by_score(monkeypatch):
    chunks = [_chunk(1, "a"), _chunk(2, "b"), _chunk(3, "c")]
    monkeypatch.setattr(reranking, "get_client", lambda: FakeClient("[1, 9, 5]"))
    ranked = reranking.rerank("q", chunks, 2)
    assert [c.chunk_id for c in ranked] == [2, 3]
    assert ranked[0].score == 9.0


def test_all_zero_scores_fall_back_and_warn(monkeypatch, caplog):
    chunks = [_chunk(1, "a"), _chunk(2, "b")]
    monkeypatch.setattr(reranking, "get_client", lambda: FakeClient("[0, 0]"))
    with caplog.at_level(logging.WARNING, logger=reranking.logger.name):
        ranked = reranking.rerank("q", chunks, 2)
    assert [c.chunk_id for c in ranked] == [1, 2]
    assert "scored every passage 0" in caplog.text


def test_fallbacks_are_counted(monkeypatch):
    chunks = [_chunk(1, "a")]
    before = reranking.stats["fallbacks"]
    monkeypatch.setattr(reranking, "get_client", lambda: FakeClient("[0]"))
    reranking.rerank("q", chunks, 1)
    assert reranking.stats["fallbacks"] == before + 1
    monkeypatch.setattr(reranking, "get_client", lambda: FakeClient("[7]"))
    reranking.rerank("q", chunks, 1)
    assert reranking.stats["fallbacks"] == before + 1
