"""Unit tests for the chunker — pure, no network or DB required."""
import pytest

from app.services.chunking import chunk_markdown, chunk_text, split_markdown_sections


def test_short_text_returns_single_chunk():
    assert chunk_text("hello world", chunk_size=100, chunk_overlap=10) == ["hello world"]


def test_empty_or_whitespace_returns_no_chunks():
    assert chunk_text("   \n  ", chunk_size=100, chunk_overlap=10) == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("a b c d", chunk_size=10, chunk_overlap=10)


def test_long_text_splits_into_multiple_chunks():
    text = " ".join(str(i) for i in range(3000))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunks_overlap_preserves_context():
    text = " ".join(str(i) for i in range(500))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=40)
    # With overlap, consecutive chunks should share some content.
    assert len(chunks) >= 2


def test_markdown_sections_split_at_headings():
    text = "# A\n\nalpha\n\n## B\n\nbeta\n\n### C\n\ngamma"
    assert len(split_markdown_sections(text)) == 3


def test_markdown_ignores_headings_inside_code_fences():
    text = "# Title\n\n```python\n# not a heading\nprint(1)\n```\n\n# Next\n\ntext"
    sections = split_markdown_sections(text)
    assert len(sections) == 2
    assert "# not a heading" in sections[0]


def test_markdown_small_sections_are_merged():
    chunks = chunk_markdown("# A\n\nhi\n\n# B\n\nyo", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert "# A" in chunks[0] and "# B" in chunks[0]


def test_markdown_chunks_start_at_heading_boundaries():
    body = " ".join(["word"] * 30)
    text = "\n\n".join(f"# Section {i}\n\n{body}" for i in range(3))
    chunks = chunk_markdown(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) == 3
    assert all(c.startswith("# Section") for c in chunks)


def test_markdown_long_section_repeats_heading():
    text = "## Big\n\n" + " ".join(str(i) for i in range(400))
    chunks = chunk_markdown(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.startswith("## Big") for c in chunks)


def test_markdown_empty_returns_no_chunks():
    assert chunk_markdown("  \n ", chunk_size=100, chunk_overlap=10) == []
