from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from invarlock.eval.providers.base import (
    deterministic_shards,
    deterministic_worker_init_fn,
)


def test_deterministic_worker_init_fn_handles_seed_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import random

    def bad_seed(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("seed failure")

    def bad_np_seed(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("numpy seed failure")

    monkeypatch.setattr(random, "seed", bad_seed, raising=False)
    monkeypatch.setattr(np.random, "seed", bad_np_seed, raising=False)

    # Should not raise even if seeding fails
    deterministic_worker_init_fn(worker_id=1, base_seed=42)


def test_deterministic_worker_init_fn_uses_torch_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, list[int]] = {"seed": [], "cuda_seed_all": []}

    def manual_seed(value: int) -> None:
        calls["seed"].append(value)

    class _Cuda:
        def manual_seed_all(self, value: int) -> None:
            calls["cuda_seed_all"].append(value)

    torch_stub = types.SimpleNamespace(manual_seed=manual_seed, cuda=_Cuda())

    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    deterministic_worker_init_fn(worker_id=3, base_seed=7)

    assert calls["seed"]
    assert calls["cuda_seed_all"]


def test_deterministic_shards_respects_num_workers_zero_and_many() -> None:
    all_items = list(range(10))

    shards_zero = deterministic_shards(len(all_items), num_workers=0)
    assert shards_zero == [all_items]

    shards_many = deterministic_shards(len(all_items), num_workers=4)
    flat_many = [idx for shard in shards_many for idx in shard]
    assert sorted(flat_many) == all_items
