from __future__ import annotations

from unittest import mock

from invarlock.eval.data import WikiText2Provider


def test_stratification_stats_and_scorer_profile():
    with mock.patch.object(
        WikiText2Provider, "_validate_dependencies", lambda self: None
    ):
        provider = WikiText2Provider()

    def fake_collect(self, texts, indices, tokenizer, seq_len):
        return [
            (idx, [idx % 7 + 1] * seq_len, [1] * seq_len, seq_len) for idx in indices
        ]

    with mock.patch.object(
        WikiText2Provider, "_collect_tokenized_samples", fake_collect
    ):
        texts = [f"sample {i}" for i in range(64)]
        with mock.patch.object(provider, "load", return_value=texts):
            preview, final = provider.windows(
                tokenizer=None, preview_n=8, final_n=8, seed=7
            )

    assert len(preview) == 8
    assert len(final) == 8

    stats = provider.stratification_stats
    assert stats is not None
    assert stats["pool_size"] == 16
    assert stats["batch_size_used"] >= 16
    assert "difficulty_gap" in stats

    profile = provider.scorer_profile
    assert profile is not None
    assert profile["mode"] == "byte_ngram"
    assert profile["tokens_per_second"] > 0
