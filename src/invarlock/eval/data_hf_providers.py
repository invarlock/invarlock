"""Hosted Hugging Face dataset providers."""

from __future__ import annotations

import random
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from invarlock.core.exceptions import DataError as _DataErr
from invarlock.core.exceptions import ValidationError as _ValErr

from .data_support import (
    EvaluationWindow,
    _require_load_dataset,
    load_dataset_with_cache_fallback,
)
from .data_tokenization import tokenize_combined_pairs, tokenize_texts_padded


def _facade_attr(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("invarlock.eval.data_providers")
    if facade is None:
        return fallback
    return getattr(facade, name, fallback)


def _require_dataset(message: str) -> None:
    require_fn = _facade_attr("_require_load_dataset", _require_load_dataset)
    require_fn(message)


def _load_dataset_with_cache_fallback(*args: Any, **kwargs: Any) -> Any:
    load_fn = _facade_attr(
        "load_dataset_with_cache_fallback", load_dataset_with_cache_fallback
    )
    return load_fn(*args, **kwargs)


def _field_value(row: Mapping[str, Any], field: str) -> Any:
    if not field:
        return None
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


class HFTextProvider:
    name = "hf_text"

    def __init__(
        self,
        dataset_name: str | None = None,
        config_name: str | None = None,
        text_field: str = "text",
        cache_dir: str | None = None,
        trust_remote_code: bool = False,
        max_samples: int = 2000,
    ):
        _require_dataset(
            "DEPENDENCY-MISSING: datasets library required for hf_text provider"
        )
        self.dataset_name = dataset_name or "Salesforce/wikitext"
        self.config_name = config_name or None
        self.text_field = text_field
        self.cache_dir = cache_dir
        self.trust_remote_code = bool(trust_remote_code)
        self.max_samples = int(max_samples)
        self._texts_cache: dict[str, list[str]] = {}

    def load(self, split: str = "validation", **kwargs: Any) -> list[str]:
        cached = self._texts_cache.get(split)
        if cached is not None:
            return cached
        ds = _load_dataset_with_cache_fallback(
            path=self.dataset_name,
            name=self.config_name,
            split=split,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        texts: list[str] = []
        count = 0
        for row in ds:
            if self.text_field not in row:
                continue
            value = row[self.text_field]
            if isinstance(value, str) and value.strip():
                texts.append(value)
                count += 1
                if count >= self.max_samples:
                    break
        self._texts_cache[split] = texts
        return texts

    def _simple_tokenize(
        self, texts: list[str], tokenizer: Any, seq_len: int, indices: list[int]
    ) -> EvaluationWindow:
        input_ids_list, attention_masks_list, valid_indices = tokenize_texts_padded(
            texts,
            tokenizer,
            seq_len,
            positions=indices,
        )
        return EvaluationWindow(
            input_ids_list,
            attention_masks_list,
            valid_indices,
        )

    def _token_signature(
        self, input_ids: Sequence[int], attention_mask: Sequence[int]
    ) -> tuple[int, ...]:
        return tuple(
            int(token_id)
            for token_id, mask in zip(input_ids, attention_mask, strict=False)
            if int(mask)
        )

    def _collect_unique_window_samples(
        self,
        texts: Sequence[str],
        tokenizer: Any,
        *,
        seq_len: int,
        positions: Sequence[int],
        target_total: int,
    ) -> list[tuple[int, list[int], list[int]]]:
        unique_samples: list[tuple[int, list[int], list[int]]] = []
        seen_signatures: set[tuple[int, ...]] = set()
        chunk_size = max(64, min(256, int(target_total or 1)))
        for start in range(0, len(positions), chunk_size):
            batch_positions = [
                int(pos) for pos in positions[start : start + chunk_size]
            ]
            batch_texts = [texts[pos] for pos in batch_positions]
            tokenized_window = self._simple_tokenize(
                batch_texts,
                tokenizer,
                seq_len,
                batch_positions,
            )
            for idx, input_ids, attention_mask in zip(
                tokenized_window.indices,
                tokenized_window.input_ids,
                tokenized_window.attention_masks,
                strict=False,
            ):
                signature = self._token_signature(input_ids, attention_mask)
                if len(signature) <= 1 or signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                unique_samples.append(
                    (
                        int(idx),
                        [int(token) for token in input_ids],
                        [int(mask) for mask in attention_mask],
                    )
                )
                if len(unique_samples) >= target_total:
                    return unique_samples
        return unique_samples

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
        texts = self.load(split=split)
        if not texts:
            raise _DataErr(
                code="E306",
                message=(
                    "NO-SAMPLES: hf_text produced no samples; check dataset_name/config_name/text_field"
                ),
            )
        total_required = int(preview_n) + int(final_n)
        if total_required <= 0:
            raise _ValErr(
                code="E302",
                message="VALIDATION-FAILED: preview/final must be positive",
            )
        selected_positions = list(range(len(texts)))
        random_mod = _facade_attr("random", random)
        random_mod.Random(int(seed)).shuffle(selected_positions)
        unique_samples = self._collect_unique_window_samples(
            texts,
            tokenizer,
            seq_len=seq_len,
            positions=selected_positions,
            target_total=total_required,
        )
        if len(unique_samples) < total_required:
            raise _DataErr(
                code="E304",
                message=(
                    "TOKENIZE-INSUFFICIENT: failed to gather enough unique tokenized samples"
                ),
                details={
                    "needed": int(total_required),
                    "got": int(len(unique_samples)),
                },
            )
        preview_samples = unique_samples[: int(preview_n)]
        final_samples = unique_samples[int(preview_n) : total_required]
        return (
            EvaluationWindow(
                input_ids=[sample[1] for sample in preview_samples],
                attention_masks=[sample[2] for sample in preview_samples],
                indices=[sample[0] for sample in preview_samples],
            ),
            EvaluationWindow(
                input_ids=[sample[1] for sample in final_samples],
                attention_masks=[sample[2] for sample in final_samples],
                indices=[sample[0] for sample in final_samples],
            ),
        )

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
        texts = self.load(split=split)
        return {
            "total_tokens": 0,
            "available_nonoverlap": len(texts),
            "available_unique": len(texts),
            "dedupe_rate": 0.0,
            "stride": stride,
            "seq_len": seq_len,
            "candidate_unique": len(texts),
            "candidate_limit": min(len(texts), self.max_samples),
        }


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
        _require_dataset(
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
            EvaluationWindow(preview_input_ids, preview_attention_masks, preview_indices),
            EvaluationWindow(final_input_ids, final_attention_masks, final_indices),
            preview_labels,
            final_labels,
        )

    def _load_pairs(self, split: str) -> list[tuple[str, str]]:
        cached = self._pairs_cache.get(split)
        if cached is not None:
            return cached
        ds = _load_dataset_with_cache_fallback(
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
            src = _field_value(row, self.src_field)
            tgt = _field_value(row, self.tgt_field)
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


__all__ = ["HFSeq2SeqProvider", "HFTextProvider"]
