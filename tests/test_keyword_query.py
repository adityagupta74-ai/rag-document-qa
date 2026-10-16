"""Unit tests for the keyword-query builder — pure, no network or DB."""
from app.services.retrieval import build_keyword_query


def test_question_words_are_or_joined():
    assert build_keyword_query("How many vacation days?") == "How or many or vacation or days"


def test_literal_or_is_dropped_so_no_dangling_operators():
    assert build_keyword_query("or cats or") == "cats"


def test_punctuation_and_hyphens_are_split():
    assert build_keyword_query("What is AES-256?") == "What or is or AES or 256"


def test_empty_question_gives_empty_query():
    assert build_keyword_query("  ?! ") == ""
