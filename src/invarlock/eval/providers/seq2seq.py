"""
Seq2Seq provider (Phase 2 scaffold).

Future implementation will stream paired (encoder_inputs, decoder_labels) with
stable example IDs and a digest of tokenization/EOS policies.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..data_support import EventEmitter
from ..data_windows import EvaluationWindow, split_window_by_index
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

    name = "seq2seq"

    def __init__(self, emit: EventEmitter | None = None, **kwargs: Any) -> None:
        self._n = int(kwargs.get("n", 12))
        self._src_len = int(kwargs.get("src_len", 6))
        self._tgt_len = int(kwargs.get("tgt_len", 7))
        self._pad_id = int(kwargs.get("pad_id", 0))
        self._bos_id = int(kwargs.get("bos_id", 1))
        self._eos_id = int(kwargs.get("eos_id", 2))
        self._emit_event = emit
        self._ids: list[str] = []
        self.last_preview_labels: list[list[int]] | None = None
        self.last_final_labels: list[list[int]] | None = None

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

    def load(
        self, split: str = "validation", **kwargs: Any
    ) -> list[str]:  # pragma: no cover - seq2seq does not expose raw-text windows
        _ = split, kwargs
        return []

    def windows(
        self,
        tokenizer: Any,
        *,
        seq_len: int = 128,
        stride: int = 64,
        preview_n: int = 100,
        final_n: int = 100,
        seed: int = 42,
        split: str = "validation",
    ) -> tuple[EvaluationWindow, EvaluationWindow]:
        _ = stride, split
        total = max(1, int(preview_n) + int(final_n))
        if self._n < total:
            self._n = total
        batches = list(self.batches(seed=seed, batch_size=total))
        if not batches:
            raise ValueError("seq2seq provider produced no examples")
        batch = batches[0]
        src_ids_list = [list(x) for x in batch.get("src_ids", [])][:total]
        src_mask_list = [list(x) for x in batch.get("src_mask", [])][:total]
        tgt_ids_list = [list(x) for x in batch.get("tgt_ids", [])][:total]
        pad_id = getattr(tokenizer, "pad_token_id", self._pad_id)

        def _pad(seq: list[int], *, fill: int) -> list[int]:
            if len(seq) < seq_len:
                return (seq + [fill] * (seq_len - len(seq)))[:seq_len]
            return seq[:seq_len]

        input_ids = [_pad(seq, fill=pad_id) for seq in src_ids_list]
        attention_masks: list[list[int]] = []
        for index, seq in enumerate(input_ids):
            if index < len(src_mask_list) and len(src_mask_list[index]) == len(
                src_ids_list[index]
            ):
                attention_masks.append(
                    [int(v) for v in _pad(list(src_mask_list[index]), fill=0)]
                )
            else:
                attention_masks.append([1 if token != pad_id else 0 for token in seq])

        preview_window, final_window = split_window_by_index(
            EvaluationWindow(
                input_ids=input_ids,
                attention_masks=attention_masks,
                indices=list(range(len(input_ids))),
            ),
            split_index=preview_n,
        )
        self.last_preview_labels = [
            _pad(list(seq), fill=-100) for seq in tgt_ids_list[:preview_n]
        ]
        self.last_final_labels = [
            _pad(list(seq), fill=-100)
            for seq in tgt_ids_list[preview_n : preview_n + final_n]
        ]
        return preview_window, final_window

    def estimate_capacity(
        self,
        tokenizer: Any,
        *,
        seq_len: int,
        stride: int,
        split: str = "validation",
        target_total: int | None = None,
        fast_mode: bool = False,
    ) -> dict[str, Any]:
        _ = tokenizer, split, target_total, fast_mode
        return {
            "total_tokens": int(self._n * seq_len),
            "available_nonoverlap": int(self._n),
            "available_unique": int(self._n),
            "dedupe_rate": 0.0,
            "stride": int(stride),
            "seq_len": int(seq_len),
            "candidate_unique": int(self._n),
            "candidate_limit": int(self._n),
            "tokens_available": int(self._n * seq_len),
            "examples_available": int(self._n),
        }
