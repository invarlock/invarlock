from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.eval.data_capacity import estimate_wikitext2_capacity
from invarlock.eval.data_difficulty import score_candidates_byte_ngram
from invarlock.eval.data_stratification import stratify_wikitext_candidates


def test_wikitext2_capacity_helper_fast_and_slow_paths(monkeypatch):
    calls: list[tuple[str, object]] = []

    def load_fn(*, split: str, max_samples: int):
        calls.append(("load", (split, max_samples)))
        return ["alpha", "beta", "gamma", "delta"]

    def collect_fn(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        calls.append(("collect", tuple(indices)))
        return [
            (idx, [idx + 1, idx + 2], [1, 1], 2)
            for idx in indices
        ]

    monkeypatch.setenv("INVARLOCK_CAPACITY_FAST", "1")
    fast = estimate_wikitext2_capacity(
        load_fn=load_fn,
        collect_tokenized_samples_fn=collect_fn,
        tokenizer=SimpleNamespace(),
        seq_len=8,
        stride=4,
    )
    assert fast["available_nonoverlap"] == 4
    assert fast["candidate_unique"] == 4
    assert calls == [("load", ("validation", 2000))]

    monkeypatch.delenv("INVARLOCK_CAPACITY_FAST", raising=False)
    calls.clear()
    slow = estimate_wikitext2_capacity(
        load_fn=load_fn,
        collect_tokenized_samples_fn=collect_fn,
        tokenizer=SimpleNamespace(),
        seq_len=8,
        stride=4,
        target_total=2,
    )
    assert slow["available_nonoverlap"] == 4
    assert slow["candidate_limit"] == 4
    assert slow["candidate_unique"] == 4
    assert calls[0] == ("load", ("validation", 2000))
    assert calls[1] == ("collect", (0, 1, 2, 3))
    assert calls[2] == ("collect", (0, 1, 2, 3))


def test_byte_ngram_helper_is_deterministic_and_mutates_candidates():
    candidates = [{"text": "alpha"}, {"text": None}]
    profile = score_candidates_byte_ngram(
        candidates,
        order=4,
        pad_token=256,
        alpha=1.0,
    )
    assert profile is not None
    assert profile["mode"] == "byte_ngram"
    assert all("difficulty" in candidate for candidate in candidates)

    candidates_second = [{"text": "alpha"}, {"text": None}]
    profile_second = score_candidates_byte_ngram(
        candidates_second,
        order=4,
        pad_token=256,
        alpha=1.0,
    )
    assert profile_second is not None
    assert [candidate["difficulty"] for candidate in candidates] == pytest.approx(
        [candidate["difficulty"] for candidate in candidates_second], rel=0, abs=0
    )


def test_stratification_helper_builds_balanced_windows():
    candidates = [
        {
            "dataset_index": idx,
            "difficulty": float(idx),
            "input_ids": [idx, idx + 1],
            "attention_mask": [1, 1],
        }
        for idx in range(8)
    ]
    preview, final, stats = stratify_wikitext_candidates(
        candidates,
        preview_n=3,
        final_n=3,
        reserve=2,
        batch_size_used=8,
    )
    assert len(preview) == 3
    assert len(final) == 3
    assert stats["pool_size"] == 6
    assert stats["batch_size_used"] == 8
    assert "difficulty_gap" in stats

