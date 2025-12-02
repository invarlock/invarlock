"""
EvaluationProvider base Protocol (Phase 1 scaffold).

Providers encapsulate dataset/task specifics (pairing, masking, transforms),
exposing a stable schedule and a reproducibility digest so metrics can be
computed in a task-agnostic way.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol


def deterministic_worker_init_fn(worker_id: int, *, base_seed: int = 0) -> None:
    """Best-effort deterministic worker initializer.

    Sets RNG seeds for `random`, `numpy`, and `torch` (if available) using a
    stable derivation from `base_seed` and `worker_id`.
    """
    try:
        import random

        random.seed((base_seed ^ (worker_id + 17)) & 0x7FFFFFFF)
    except Exception:
        pass
    try:
        import numpy as _np

        _np.random.seed(((base_seed + 97) ^ (worker_id * 131)) & 0x7FFFFFFF)
    except Exception:
        pass
    try:  # pragma: no cover - torch may be unavailable in CI
        import torch as _torch

        _torch.manual_seed((base_seed * 1009 + worker_id * 7919) & 0x7FFFFFFF)
        if hasattr(_torch.cuda, "manual_seed_all"):
            _torch.cuda.manual_seed_all(
                (base_seed * 1013 + worker_id * 7951) & 0x7FFFFFFF
            )
    except Exception:
        pass


def deterministic_shards(n: int, *, num_workers: int) -> list[list[int]]:
    """Return a deterministic partition of `range(n)` across `num_workers` workers.

    Uses simple modulo-based assignment to ensure stable sharding independent of
    process scheduling. `num_workers <= 1` yields a single shard with all items.
    """
    if num_workers is None or num_workers <= 1:
        return [list(range(n))]
    shards: list[list[int]] = [[] for _ in range(int(num_workers))]
    for i in range(int(n)):
        shards[i % int(num_workers)].append(i)
    return shards


class EvaluationProvider(Protocol):
    def pairing_schedule(self) -> list[str]:
        """Return a stable, sorted list of example IDs used for pairing."""

    def digest(self) -> dict[str, Any]:
        """Return a reproducibility digest (tokenizer/masking/transform hashes)."""

    def batches(self, *, seed: int, batch_size: int) -> Iterable[dict[str, Any]]:
        """Yield task-appropriate batches (input tensors/labels/masks)."""
