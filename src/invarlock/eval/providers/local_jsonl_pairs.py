from __future__ import annotations

from pathlib import Path
from typing import Any

from ..data_tokenization import tokenize_combined_pairs
from ..data_windows import (
    EvaluationWindow,
    split_labels_by_index,
    split_window_by_index,
)
from .local_jsonl_shared import (
    load_local_jsonl_pairs,
    local_jsonl_cache_key,
    resolve_local_jsonl_files,
)


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


__all__ = ["LocalJSONLPairsProvider"]
