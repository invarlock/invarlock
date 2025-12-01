from __future__ import annotations

import types
from unittest import mock

import pytest
import torch

from invarlock.eval.data import WikiText2Provider


class _DummyModel(torch.nn.Module):
    """Simple deterministic model for scoring tests."""

    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self._device = torch.device("cpu")

    def eval(self):
        return self

    def to(self, device):
        self._device = torch.device(device)
        return self

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(
            batch,
            seq_len,
            self.vocab_size,
            device=self._device,
            dtype=torch.float32,
        )
        for b in range(batch):
            for t in range(seq_len):
                token = int(input_ids[b, t].item()) % self.vocab_size
                logits[b, t, token] = 2.0
                logits[b, t, (token + 1) % self.vocab_size] = 1.0
        return types.SimpleNamespace(logits=logits)


def test_batched_scorer_matches_per_example():
    with mock.patch.object(
        WikiText2Provider, "_validate_dependencies", lambda self: None
    ):
        provider = WikiText2Provider()

    dummy_model = _DummyModel()
    provider._difficulty_model = dummy_model
    provider._difficulty_device = torch.device("cpu")
    provider.__class__._MODEL_CACHE = dummy_model
    provider.__class__._MODEL_DEVICE = torch.device("cpu")

    candidates = [
        {
            "dataset_index": idx,
            "input_ids": [idx % 5 + 1, idx % 7 + 2, idx % 11 + 3, idx % 13 + 4],
            "attention_mask": [1, 1, 1, 1],
            "token_count": 4,
        }
        for idx in range(12)
    ]

    try:
        provider._score_candidates_with_model(candidates)

        batched_losses = [cand["difficulty"] for cand in candidates]

        manual_losses = []
        with torch.no_grad():
            for cand in candidates:
                inp = torch.tensor(cand["input_ids"], dtype=torch.long).unsqueeze(0)
                attn = torch.tensor(cand["attention_mask"], dtype=torch.long).unsqueeze(
                    0
                )
                outputs = dummy_model(inp, attention_mask=attn)
                shift_logits = outputs.logits[:, :-1, :]
                shift_labels = inp[:, 1:]
                shift_mask = attn[:, 1:]
                losses = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, dummy_model.vocab_size),
                    shift_labels.reshape(-1),
                    reduction="none",
                )
                losses = losses.reshape(shift_labels.shape) * shift_mask
                token_counts = shift_mask.sum(dim=1).clamp(min=1)
                loss_value = (losses.sum(dim=1) / token_counts).item()
                manual_losses.append(loss_value)

        diffs = [
            abs(a - b) for a, b in zip(batched_losses, manual_losses, strict=False)
        ]
        assert max(diffs) <= 1e-6

        # Determinism test
        candidates_second = [cand.copy() for cand in candidates]
        provider._score_candidates_with_model(candidates_second)
        second_losses = [cand["difficulty"] for cand in candidates_second]
        assert second_losses == pytest.approx(batched_losses, rel=0, abs=0)
    finally:
        provider.__class__._cleanup_model_cache()


def test_scorer_cache_cleanup(recwarn):
    with mock.patch.object(
        WikiText2Provider, "_validate_dependencies", lambda self: None
    ):
        provider = WikiText2Provider()

    dummy_model = _DummyModel()
    provider._difficulty_model = dummy_model
    provider._difficulty_device = torch.device("cpu")
    WikiText2Provider._MODEL_CACHE = dummy_model
    WikiText2Provider._MODEL_DEVICE = torch.device("cpu")

    WikiText2Provider._cleanup_model_cache()

    assert not recwarn.list, "Cache cleanup should not emit warnings"

    assert WikiText2Provider._MODEL_CACHE is None
    assert WikiText2Provider._MODEL_DEVICE is None


def test_stratification_stats_and_scorer_profile():
    with mock.patch.object(
        WikiText2Provider, "_validate_dependencies", lambda self: None
    ):
        provider = WikiText2Provider()

    def fake_collect(self, texts, indices, tokenizer, seq_len):
        return [
            (idx, [idx % 7 + 1] * seq_len, [1] * seq_len, seq_len) for idx in indices
        ]

    def fake_score(self, candidates):
        for idx, cand in enumerate(candidates):
            cand["difficulty"] = float(idx % 5)
        self._last_batch_size_used = len(candidates)
        self._last_scorer_profile = {
            "batch_size": len(candidates),
            "tokens_processed": len(candidates) * 4,
            "elapsed_seconds": 0.5,
            "tokens_per_second": float(len(candidates) * 8),
        }
        return True

    with mock.patch.object(
        WikiText2Provider, "_collect_tokenized_samples", fake_collect
    ):
        with mock.patch.object(
            WikiText2Provider, "_score_candidates_with_model", fake_score
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
    assert profile["batch_size"] >= 16
    assert profile["tokens_per_second"] > 0
