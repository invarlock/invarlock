"""
Text LM provider (Phase 2 scaffold).

This module will house a provider that encapsulates dataset/tokenizer logic for
language modeling tasks. For now, it serves as a placeholder aligning with the
EvaluationProvider protocol.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import EvaluationProvider


class TextLMProvider(EvaluationProvider):
    """Deterministic synthetic text LM provider for tests and smokes.

    Args (kwargs):
        task: 'causal' or 'mlm' (default: 'causal')
        n: number of examples (default: 16)
        seq_len: length of each sequence (default: 8)
        mask_prob: MLM mask probability (default: 0.15)
        pad_id: padding token id (default: 0)
        eos_id: end-of-sequence id (default: 2)
    """

    def __init__(self, **kwargs: Any) -> None:
        self._task = str(kwargs.get("task", "causal")).lower()
        self._n = int(kwargs.get("n", 16))
        self._seq_len = int(kwargs.get("seq_len", 8))
        self._mask_prob = float(kwargs.get("mask_prob", 0.15))
        self._pad_id = int(kwargs.get("pad_id", 0))
        self._eos_id = int(kwargs.get("eos_id", 2))
        self._ids: list[str] = []

    def pairing_schedule(self) -> list[str]:
        return (
            sorted(self._ids) if self._ids else [f"ex{i:04d}" for i in range(self._n)]
        )

    def digest(self) -> dict[str, Any]:
        return {"provider": "text_lm", "version": 1, "task": self._task}

    def _gen_example(self, idx: int, *, seed: int) -> dict[str, Any]:
        import random

        rng = random.Random((seed + 31) ^ (idx * 131))
        # Generate a simple pattern with EOS at end and some pads
        real_len = max(3, self._seq_len - (idx % 3))
        ids = [1 + (rng.randint(0, 19)) for _ in range(real_len - 1)] + [self._eos_id]
        if real_len < self._seq_len:
            ids = ids + [self._pad_id] * (self._seq_len - real_len)
        attn = [1 if t != self._pad_id else 0 for t in ids]
        ex_id = f"ex{idx:04d}"

        labels: list[int] | None = None
        weights = sum(attn)
        if self._task == "mlm":
            labels = [-100] * len(ids)
            masked = 0
            for pos, (tok, m) in enumerate(zip(ids, attn, strict=False)):
                if not m or tok in (self._pad_id, self._eos_id):
                    continue
                # Deterministic mask using rng per-position
                rng2 = random.Random((seed + idx * 17 + pos * 13) & 0x7FFFFFFF)
                if rng2.random() < self._mask_prob:
                    labels[pos] = tok
                    masked += 1
            if masked == 0:
                # Ensure at least one label to avoid degenerate windows
                for pos, (tok, m) in enumerate(zip(ids, attn, strict=False)):
                    if m and tok not in (self._pad_id, self._eos_id):
                        labels[pos] = tok
                        masked = 1
                        break
            weights = masked

        return {
            "ids": ex_id,
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels if labels is not None else [],
            "weights": int(weights),
        }

    def batches(self, *, seed: int, batch_size: int) -> Iterable[dict[str, Any]]:
        assert batch_size > 0
        batch: dict[str, Any] = {
            "ids": [],
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "weights": [],
        }
        self._ids = []
        for i in range(self._n):
            ex = self._gen_example(i, seed=seed)
            self._ids.append(ex["ids"])
            for k in ("ids", "input_ids", "attention_mask", "labels", "weights"):
                batch[k].append(ex[k])
            if len(batch["ids"]) >= batch_size:
                yield batch
                batch = {
                    "ids": [],
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "weights": [],
                }
        if batch["ids"]:
            yield batch
