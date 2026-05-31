from __future__ import annotations

import random

import pytest

from invarlock.eval.data import TextLMProvider
from invarlock.eval.data_support import (
    deterministic_shards,
    deterministic_worker_init_fn,
)


def test_deterministic_shards_and_schedule_parity():
    n = 32
    # Simulate worker shards
    for k in (0, 1, 2, 4):
        shards = deterministic_shards(n, num_workers=k)
        # Flatten order preserving relative index order per shard; then sort to compare schedules
        flat_ids = [f"ex{i:04d}" for shard in shards for i in shard]
        combined_sorted = sorted(flat_ids)
        # Provider schedule should be stable and independent of workers
        provider = TextLMProvider(n=n)
        sched = provider.pairing_schedule()
        assert combined_sorted == sorted(sched)


def test_deterministic_worker_init_fn_reproducible():
    # Reproducible RNG states across invocations
    deterministic_worker_init_fn(0, base_seed=123)
    a = (random.random(), random.randint(0, 1000))
    deterministic_worker_init_fn(0, base_seed=123)
    b = (random.random(), random.randint(0, 1000))
    assert a == b


def test_provider_digest_independent_of_workers():
    p = TextLMProvider(n=8)
    base = p.digest()
    # Pretend to vary workers; digest should remain the same
    for k in (0, 2, 4):
        _ = deterministic_shards(8, num_workers=k)
        assert p.digest() == base


def test_text_lm_provider_validates_shape_parameters():
    with pytest.raises(ValueError, match="task"):
        TextLMProvider(task="classification")
    with pytest.raises(ValueError, match="n"):
        TextLMProvider(n=-1)
    with pytest.raises(ValueError, match="seq_len"):
        TextLMProvider(seq_len=2)
    with pytest.raises(ValueError, match="mask_prob"):
        TextLMProvider(mask_prob=float("nan"))
    with pytest.raises(ValueError, match="mask_prob"):
        TextLMProvider(mask_prob=-0.1)
    with pytest.raises(ValueError, match="mask_prob"):
        TextLMProvider(mask_prob=1.1)


def test_text_lm_provider_rejects_non_positive_batch_size():
    provider = TextLMProvider(n=1)
    with pytest.raises(ValueError, match="batch_size"):
        list(provider.batches(seed=0, batch_size=0))


def test_text_lm_provider_emits_requested_sequence_length():
    provider = TextLMProvider(n=3, seq_len=3)
    batch = next(iter(provider.batches(seed=0, batch_size=3)))
    assert [len(row) for row in batch["input_ids"]] == [3, 3, 3]
    assert [len(row) for row in batch["attention_mask"]] == [3, 3, 3]
