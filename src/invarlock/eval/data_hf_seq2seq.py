"""Hosted Hugging Face seq2seq dataset provider."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from invarlock.core.exceptions import DataError as _DataErr

from .data_hf_common import field_value, load_dataset_from_facade, require_dataset
from .data_support import EvaluationWindow
from .data_tokenization import tokenize_combined_pairs


class HFSeq2SeqProvider:
    name = "hf_seq2seq"

    def __init__(
        self,
        dataset_name: str,
        config_name: str | None = None,
        src_field: str = "source",
        tgt_field: str = "target",
        cache_dir: str | None = None,
        revision: str | None = None,
        src_prefix: str = "",
        tgt_prefix: str = "",
        max_samples: int = 2000,
    ) -> None:
        require_dataset(
            "DEPENDENCY-MISSING: datasets library required for hf_seq2seq provider"
        )
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.cache_dir = cache_dir
        self.revision = revision or None
        self.src_prefix = str(src_prefix or "")
        self.tgt_prefix = str(tgt_prefix or "")
        self.max_samples = int(max_samples)
        self.last_preview_labels: list[list[int]] | None = None
        self.last_final_labels: list[list[int]] | None = None
        self._pairs_cache: dict[str, list[tuple[str, str]]] = {}

    def _split_by_preview_positions(
        self,
        window: EvaluationWindow,
        labels: list[list[int]],
        *,
        preview_positions: Sequence[int],
    ) -> tuple[EvaluationWindow, EvaluationWindow, list[list[int]], list[list[int]]]:
        preview_lookup = {int(position) for position in preview_positions}
        preview_input_ids: list[list[int]] = []
        preview_attention_masks: list[list[int]] = []
        preview_indices: list[int] = []
        preview_labels: list[list[int]] = []
        final_input_ids: list[list[int]] = []
        final_attention_masks: list[list[int]] = []
        final_indices: list[int] = []
        final_labels: list[list[int]] = []

        for input_ids, attention_mask, index, label in zip(
            window.input_ids,
            window.attention_masks,
            window.indices,
            labels,
            strict=False,
        ):
            if int(index) in preview_lookup:
                preview_input_ids.append(input_ids)
                preview_attention_masks.append(attention_mask)
                preview_indices.append(int(index))
                preview_labels.append(label)
            else:
                final_input_ids.append(input_ids)
                final_attention_masks.append(attention_mask)
                final_indices.append(int(index))
                final_labels.append(label)

        return (
            EvaluationWindow(
                preview_input_ids, preview_attention_masks, preview_indices
            ),
            EvaluationWindow(final_input_ids, final_attention_masks, final_indices),
            preview_labels,
            final_labels,
        )

    def _load_pairs(self, split: str) -> list[tuple[str, str]]:
        cached = self._pairs_cache.get(split)
        if cached is not None:
            return cached
        ds = load_dataset_from_facade(
            path=self.dataset_name,
            name=self.config_name,
            split=split,
            cache_dir=self.cache_dir,
            revision=self.revision,
        )
        out: list[tuple[str, str]] = []
        count = 0
        for row in ds:
            if not isinstance(row, Mapping):
                continue
            src = field_value(row, self.src_field)
            tgt = field_value(row, self.tgt_field)
            if (
                isinstance(src, str)
                and src.strip()
                and isinstance(tgt, str)
                and tgt.strip()
            ):
                out.append(
                    (
                        f"{self.src_prefix}{src.strip()}",
                        f"{self.tgt_prefix}{tgt.strip()}",
                    )
                )
                count += 1
                if count >= self.max_samples:
                    break
        self._pairs_cache[split] = out
        return out

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
        pairs = self._load_pairs(split)
        if not pairs:
            raise _DataErr(
                code="E307",
                message=(
                    "NO-PAIRS: hf_seq2seq produced no pairs; check src_field/tgt_field"
                ),
            )
        indexed_pairs = list(enumerate(pairs))
        random.Random(int(seed)).shuffle(indexed_pairs)
        prev_pairs = indexed_pairs[:preview_n]
        fin_pairs = indexed_pairs[preview_n : preview_n + final_n]
        combined_pairs = [pair for _, pair in prev_pairs + fin_pairs]
        combined_positions = [position for position, _pair in prev_pairs + fin_pairs]
        combined_window, combined_labels = tokenize_combined_pairs(
            combined_pairs,
            tokenizer=tokenizer,
            seq_len=seq_len,
            positions=combined_positions,
        )
        (
            preview_window,
            final_window,
            self.last_preview_labels,
            self.last_final_labels,
        ) = self._split_by_preview_positions(
            combined_window,
            combined_labels,
            preview_positions=[position for position, _pair in prev_pairs],
        )
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
        pairs = self._load_pairs(split)
        n = len(pairs)
        return {
            "total_tokens": int(n * seq_len),
            "available_nonoverlap": n,
            "available_unique": n,
            "dedupe_rate": 0.0,
            "stride": stride,
            "seq_len": seq_len,
            "candidate_unique": n,
            "candidate_limit": n,
            "tokens_available": int(n * seq_len),
            "examples_available": n,
        }
