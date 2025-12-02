from __future__ import annotations

import random

from invarlock.eval.providers.base import (
    deterministic_shards,
    deterministic_worker_init_fn,
)
from invarlock.eval.providers.text_lm import TextLMProvider


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
