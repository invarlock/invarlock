"""
Seq2Seq provider (Phase 2 scaffold).

Future implementation will stream paired (encoder_inputs, decoder_labels) with
stable example IDs and a digest of tokenization/EOS policies.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import EvaluationProvider


class Seq2SeqProvider(EvaluationProvider):
    """Deterministic synthetic seq2seq provider for tests and smokes.

    Args (kwargs):
        n: number of examples (default: 12)
        src_len: source length (default: 6)
        tgt_len: target length (default: 7)
        pad_id: pad token id (default: 0)
        bos_id: BOS id (default: 1)
        eos_id: EOS id (default: 2)
    """

    def __init__(self, **kwargs: Any) -> None:
        self._n = int(kwargs.get("n", 12))
        self._src_len = int(kwargs.get("src_len", 6))
        self._tgt_len = int(kwargs.get("tgt_len", 7))
        self._pad_id = int(kwargs.get("pad_id", 0))
        self._bos_id = int(kwargs.get("bos_id", 1))
        self._eos_id = int(kwargs.get("eos_id", 2))
        self._ids: list[str] = []

    def pairing_schedule(self) -> list[str]:
        return (
            sorted(self._ids) if self._ids else [f"ex{i:04d}" for i in range(self._n)]
        )

    def digest(self) -> dict[str, Any]:
        return {
            "provider": "seq2seq",
            "version": 1,
            "pad_id": self._pad_id,
            "eos_id": self._eos_id,
            "bos_id": self._bos_id,
        }

    def _gen_example(self, idx: int, *, seed: int) -> dict[str, Any]:
        import random

        rng = random.Random((seed + 17) ^ (idx * 97))
        # Source: BOS + tokens + PAD
        src_real = max(3, self._src_len - (idx % 2))
        src_ids = (
            [self._bos_id]
            + [1 + rng.randint(0, 19) for _ in range(src_real - 2)]
            + [self._eos_id]
        )
        if src_real < self._src_len:
            src_ids += [self._pad_id] * (self._src_len - src_real)
        src_mask = [1 if t != self._pad_id else 0 for t in src_ids]

        # Target: tokens ending with EOS and padding
        tgt_real = max(3, self._tgt_len - (idx % 3))
        tgt_ids = [1 + rng.randint(0, 19) for _ in range(tgt_real - 1)] + [self._eos_id]
        if tgt_real < self._tgt_len:
            tgt_ids += [self._pad_id] * (self._tgt_len - tgt_real)
        tgt_mask = [1 if t != self._pad_id else 0 for t in tgt_ids]

        ex_id = f"ex{idx:04d}"
        weights = sum(1 for t, m in zip(tgt_ids, tgt_mask, strict=False) if m)
        return {
            "ids": ex_id,
            "src_ids": src_ids,
            "src_mask": src_mask,
            "tgt_ids": tgt_ids,
            "tgt_mask": tgt_mask,
            "weights": int(weights),
        }

    def batches(self, *, seed: int, batch_size: int) -> Iterable[dict[str, Any]]:
        assert batch_size > 0
        batch = {
            "ids": [],
            "src_ids": [],
            "src_mask": [],
            "tgt_ids": [],
            "tgt_mask": [],
            "weights": [],
        }
        self._ids = []
        for i in range(self._n):
            ex = self._gen_example(i, seed=seed)
            self._ids.append(ex["ids"])
            for k in ("ids", "src_ids", "src_mask", "tgt_ids", "tgt_mask", "weights"):
                batch[k].append(ex[k])
            if len(batch["ids"]) >= batch_size:
                yield batch
                batch = {
                    "ids": [],
                    "src_ids": [],
                    "src_mask": [],
                    "tgt_ids": [],
                    "tgt_mask": [],
                    "weights": [],
                }
        if batch["ids"]:
            yield batch
