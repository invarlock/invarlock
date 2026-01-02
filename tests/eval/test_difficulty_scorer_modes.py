import pytest

import invarlock.eval.data as data_mod
from invarlock.eval.data import WikiText2Provider


def test_byte_ngram_scoring_is_deterministic(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    provider = WikiText2Provider()
    candidates = [
        {"text": "alpha"},
        {"text": "beta"},
    ]
    assert provider._score_candidates_byte_ngram(candidates) is True
    first = [candidate["difficulty"] for candidate in candidates]

    candidates_second = [{"text": "alpha"}, {"text": "beta"}]
    provider._score_candidates_byte_ngram(candidates_second)
    second = [candidate["difficulty"] for candidate in candidates_second]

    assert first == pytest.approx(second, rel=0, abs=0)
    profile = provider.scorer_profile
    assert profile and profile["mode"] == "byte_ngram"


def test_byte_ngram_unicode_and_missing_text(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    provider = WikiText2Provider()
    candidates = [{"text": "café"}, {"text": None}]
    assert provider._score_candidates_byte_ngram(candidates) is True
    assert all("difficulty" in candidate for candidate in candidates)
    profile = provider.scorer_profile
    assert profile
    assert profile["tokens_processed"] == len("café".encode())
