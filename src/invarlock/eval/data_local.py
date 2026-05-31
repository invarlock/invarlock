"""Local JSONL evaluation data providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock.core.exceptions import DataError as _DataErr

from .data_support import (
    EvaluationWindow,
    load_local_jsonl_pairs,
    load_local_jsonl_texts,
    local_jsonl_cache_key,
    resolve_local_jsonl_files,
    split_labels_by_index,
    split_window_by_index,
)
from .data_tokenization import tokenize_combined_pairs, tokenize_texts_padded


class LocalJSONLProvider:
    name = "local_jsonl"

    def __init__(
        self,
        file: str | None = None,
        path: str | None = None,
        data_files: str | list[str] | None = None,
        text_field: str = "text",
        max_samples: int = 2000,
    ) -> None:
        self.file = file
        self.path = path
        self.data_files = data_files
        self.text_field = text_field or "text"
        self.max_samples = int(max_samples)
        self._load_cache: tuple[tuple[Any, ...], list[str]] | None = None

    def _resolve_files(self) -> list[Path]:
        return resolve_local_jsonl_files(
            file=self.file, path=self.path, data_files=self.data_files
        )

    def load(self, split: str = "validation", **kwargs: Any) -> list[str]:
        files = self._resolve_files()
        cache_key = local_jsonl_cache_key(
            files,
            field_names=(self.text_field,),
            max_samples=self.max_samples,
        )
        if self._load_cache is not None and self._load_cache[0] == cache_key:
            return self._load_cache[1]
        texts = load_local_jsonl_texts(
            files, text_field=self.text_field, max_samples=self.max_samples
        )
        self._load_cache = (cache_key, texts)
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
                    "NO-SAMPLES: local_jsonl produced no samples; check file/path/data_files"
                ),
            )
        total = min(len(texts), int(preview_n) + int(final_n))
        combined_window = self._simple_tokenize(
            texts[:total],
            tokenizer,
            seq_len,
            list(range(total)),
        )
        return split_window_by_index(combined_window, split_index=preview_n)

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
            "candidate_limit": len(texts),
        }


class LocalJSONLPairsProvider:
    name = "local_jsonl_pairs"

    def __init__(
        self,
        file: str | None = None,
        path: str | None = None,
        data_files: str | list[str] | None = None,
        src_field: str = "source",
        tgt_field: str = "target",
        max_samples: int = 2000,
    ) -> None:
        self.file = file
        self.path = path
        self.data_files = data_files
        self.src_field = src_field or "source"
        self.tgt_field = tgt_field or "target"
        self.max_samples = int(max_samples)
        self.last_preview_labels: list[list[int]] | None = None
        self.last_final_labels: list[list[int]] | None = None
        self._pairs_cache: tuple[tuple[Any, ...], list[tuple[str, str]]] | None = None

    def _resolve_files(self) -> list[Path]:
        return resolve_local_jsonl_files(
            file=self.file, path=self.path, data_files=self.data_files
        )

    def _load_pairs(self) -> list[tuple[str, str]]:
        files = self._resolve_files()
        cache_key = local_jsonl_cache_key(
            files,
            field_names=(self.src_field, self.tgt_field),
            max_samples=self.max_samples,
        )
        if self._pairs_cache is not None and self._pairs_cache[0] == cache_key:
            return self._pairs_cache[1]
        pairs = load_local_jsonl_pairs(
            files,
            src_field=self.src_field,
            tgt_field=self.tgt_field,
            max_samples=self.max_samples,
        )
        self._pairs_cache = (cache_key, pairs)
        return pairs

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
        pairs = self._load_pairs()
        if not pairs:
            raise ValueError(
                "local_jsonl_pairs produced no pairs; check src_field/tgt_field and files"
            )
        prev_pairs = pairs[:preview_n]
        fin_pairs = pairs[preview_n : preview_n + final_n]
        combined_pairs = prev_pairs + fin_pairs
        combined_positions = list(range(len(prev_pairs))) + list(
            range(preview_n, preview_n + len(fin_pairs))
        )
        combined_window, combined_labels = tokenize_combined_pairs(
            combined_pairs,
            tokenizer=tokenizer,
            seq_len=seq_len,
            positions=combined_positions,
        )
        preview_window, final_window = split_window_by_index(
            combined_window, split_index=preview_n
        )
        self.last_preview_labels, self.last_final_labels = split_labels_by_index(
            combined_labels,
            combined_window.indices,
            split_index=preview_n,
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
        pairs = self._load_pairs()
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


__all__ = ["LocalJSONLProvider", "LocalJSONLPairsProvider"]
