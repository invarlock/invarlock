"""
InvarLock Evaluation Data Loading
============================

Pluggable data loading system with deterministic windowing for reproducible evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import time
import warnings
from abc import abstractmethod
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol, cast

import numpy as np

from invarlock.core.exceptions import DataError as _DataErr
from invarlock.core.exceptions import DependencyError as _DepErr
from invarlock.core.exceptions import ValidationError as _ValErr

# NOTE: During the typed-only migration, avoid hybrid KeyError mixin

_LIGHT_IMPORT = os.getenv("INVARLOCK_LIGHT_IMPORT", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

HAS_DATASETS = importlib.util.find_spec("datasets") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
_DATASETS_UNSET = object()
_load_dataset_cached: Any = _DATASETS_UNSET
load_dataset: Any = None


EventEmitter = Callable[[str, str, str | None], None]


def _get_load_dataset() -> Any | None:
    global HAS_DATASETS, _load_dataset_cached
    if callable(load_dataset):
        HAS_DATASETS = True
        return load_dataset
    if HAS_DATASETS is False:
        return None
    if _load_dataset_cached is _DATASETS_UNSET:
        try:
            from datasets import load_dataset as _datasets_load_dataset
        except ImportError:
            HAS_DATASETS = False
            _load_dataset_cached = None
        else:
            HAS_DATASETS = True
            _load_dataset_cached = _datasets_load_dataset
    return None if _load_dataset_cached is _DATASETS_UNSET else _load_dataset_cached


def _require_load_dataset(message: str) -> Any:
    load_dataset_fn = _get_load_dataset()
    if load_dataset_fn is None:
        raise _DepErr(
            code="E301",
            message=message,
            details={"dependency": "datasets"},
        )
    return load_dataset_fn


def _call_tokenizer(tokenizer: Any, /, *args: Any, **kwargs: Any) -> Any:
    return cast(Any, tokenizer)(*args, **kwargs)


def _to_python_token_rows(value: Any, *, batch_size: int) -> list[list[int]]:
    candidate = value
    if batch_size == 1 and hasattr(candidate, "squeeze"):
        try:
            candidate = candidate.squeeze(0)
        except Exception:
            pass
    if hasattr(candidate, "detach"):
        try:
            candidate = candidate.detach()
        except Exception:
            pass
    if hasattr(candidate, "cpu"):
        try:
            candidate = candidate.cpu()
        except Exception:
            pass
    if hasattr(candidate, "tolist"):
        try:
            candidate = candidate.tolist()
        except Exception:
            pass
    if batch_size == 1:
        if (
            isinstance(candidate, list)
            and candidate
            and isinstance(candidate[0], (list, tuple))
        ):
            rows = candidate[:1]
        else:
            rows = [candidate]
    else:
        if not isinstance(candidate, list) or (
            candidate and not isinstance(candidate[0], (list, tuple))
        ):
            raise TypeError("Tokenizer did not return batched rows")
        rows = candidate[:batch_size]
    return [[int(token) for token in row] for row in rows]


def _pad_token_ids_and_mask(
    token_ids: Sequence[int],
    *,
    seq_len: int,
    pad_id: int,
) -> tuple[list[int], list[int]]:
    raw_ids = [int(token) for token in token_ids[:seq_len]]
    real_tokens = len(raw_ids)
    if real_tokens < seq_len:
        raw_ids.extend([pad_id] * (seq_len - real_tokens))
    attention_mask = [1] * real_tokens
    if real_tokens < seq_len:
        attention_mask.extend([0] * (seq_len - real_tokens))
    return raw_ids, attention_mask


def _call_tokenizer_compat(tokenizer: Any, text_or_texts: Any, seq_len: int) -> Any:
    option_sets = (
        {
            "truncation": True,
            "padding": "max_length",
            "max_length": seq_len,
            "return_attention_mask": False,
        },
        {
            "truncation": True,
            "padding": "max_length",
            "max_length": seq_len,
        },
        {
            "truncation": True,
            "max_length": seq_len,
        },
    )
    for kwargs in option_sets:
        try:
            return _call_tokenizer(tokenizer, text_or_texts, **kwargs)
        except TypeError:
            continue
    return _call_tokenizer(
        tokenizer,
        text_or_texts,
        truncation=True,
        max_length=seq_len,
    )


def _extract_padded_token_rows(
    tokens: Any,
    *,
    batch_size: int,
    seq_len: int,
    pad_id: int,
) -> tuple[list[list[int]], list[list[int]]]:
    token_rows = _to_python_token_rows(tokens["input_ids"], batch_size=batch_size)
    if len(token_rows) != batch_size:
        raise ValueError("Tokenizer returned unexpected row count")

    attention_value = tokens.get("attention_mask")
    attention_rows = (
        _to_python_token_rows(attention_value, batch_size=batch_size)
        if attention_value is not None
        else []
    )

    input_ids_list: list[list[int]] = []
    attention_masks_list: list[list[int]] = []
    for index, token_row in enumerate(token_rows):
        padded_ids, inferred_mask = _pad_token_ids_and_mask(
            token_row, seq_len=seq_len, pad_id=pad_id
        )
        if attention_rows:
            mask_row = [int(mask) for mask in attention_rows[index][:seq_len]]
            if len(mask_row) < seq_len:
                mask_row.extend([0] * (seq_len - len(mask_row)))
        elif len(token_row) < seq_len:
            mask_row = inferred_mask
        else:
            mask_row = [1 if token != pad_id else 0 for token in padded_ids]
        input_ids_list.append(padded_ids)
        attention_masks_list.append(mask_row)
    return input_ids_list, attention_masks_list


def _encode_text_compat(tokenizer: Any, text: str, seq_len: int) -> list[int]:
    try:
        encoded = tokenizer.encode(
            text,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
    except TypeError:
        encoded = tokenizer.encode(text, truncation=True, max_length=seq_len)
    return [int(token) for token in encoded]


def _tokenize_texts_padded(
    texts: Sequence[str],
    tokenizer: Any,
    seq_len: int,
    *,
    positions: Sequence[int] | None = None,
    warn_on_failure: bool = False,
    batch_size: int = 128,
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    if positions is None:
        positions = list(range(len(texts)))
    if len(texts) != len(positions):
        raise ValueError("texts and positions must have matching lengths")

    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    input_ids_list: list[list[int]] = []
    attention_masks_list: list[list[int]] = []
    kept_positions: list[int] = []
    use_batch_call = callable(tokenizer)
    chunk_size = max(1, min(int(batch_size), len(texts) or 1))

    for start in range(0, len(texts), chunk_size):
        stop = min(start + chunk_size, len(texts))
        chunk_texts = list(texts[start:stop])
        chunk_positions = [int(pos) for pos in positions[start:stop]]
        chunk_processed = False

        if use_batch_call:
            try:
                batch_tokens = _call_tokenizer_compat(tokenizer, chunk_texts, seq_len)
                chunk_input_ids, chunk_attention_masks = _extract_padded_token_rows(
                    batch_tokens,
                    batch_size=len(chunk_texts),
                    seq_len=seq_len,
                    pad_id=pad_id,
                )
                input_ids_list.extend(chunk_input_ids)
                attention_masks_list.extend(chunk_attention_masks)
                kept_positions.extend(chunk_positions)
                chunk_processed = True
            except Exception:
                chunk_processed = False

        if chunk_processed:
            continue

        for text, position in zip(chunk_texts, chunk_positions, strict=False):
            try:
                if hasattr(tokenizer, "encode"):
                    input_ids, attention_mask = _pad_token_ids_and_mask(
                        _encode_text_compat(tokenizer, text, seq_len),
                        seq_len=seq_len,
                        pad_id=pad_id,
                    )
                else:
                    single_tokens = _call_tokenizer_compat(tokenizer, text, seq_len)
                    token_rows, mask_rows = _extract_padded_token_rows(
                        single_tokens,
                        batch_size=1,
                        seq_len=seq_len,
                        pad_id=pad_id,
                    )
                    input_ids = token_rows[0]
                    attention_mask = mask_rows[0]
                input_ids_list.append(input_ids)
                attention_masks_list.append(attention_mask)
                kept_positions.append(position)
            except Exception as exc:
                if warn_on_failure:
                    warnings.warn(
                        f"Failed to tokenize sample {position}: {exc}",
                        stacklevel=2,
                    )

    return input_ids_list, attention_masks_list, kept_positions


class EvaluationWindow(NamedTuple):
    """A window of tokenized samples for evaluation."""

    input_ids: list[list[int]]  # List of tokenized sequences
    attention_masks: list[list[int]]  # Attention masks (1=real token, 0=padding)
    indices: list[int]  # Original dataset indices

    def __len__(self) -> int:
        return len(self.input_ids)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_ids": self.input_ids,
            "attention_masks": self.attention_masks,
            "indices": self.indices,
            "length": len(self.input_ids),
        }


def _split_window_by_index(
    window: EvaluationWindow, *, split_index: int
) -> tuple[EvaluationWindow, EvaluationWindow]:
    preview_input_ids: list[list[int]] = []
    preview_attention_masks: list[list[int]] = []
    preview_indices: list[int] = []
    final_input_ids: list[list[int]] = []
    final_attention_masks: list[list[int]] = []
    final_indices: list[int] = []

    for input_ids, attention_mask, index in zip(
        window.input_ids,
        window.attention_masks,
        window.indices,
        strict=False,
    ):
        if index < split_index:
            preview_input_ids.append(input_ids)
            preview_attention_masks.append(attention_mask)
            preview_indices.append(index)
        else:
            final_input_ids.append(input_ids)
            final_attention_masks.append(attention_mask)
            final_indices.append(index)

    return (
        EvaluationWindow(preview_input_ids, preview_attention_masks, preview_indices),
        EvaluationWindow(final_input_ids, final_attention_masks, final_indices),
    )


def _tokenize_combined_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    tokenizer: Any,
    seq_len: int,
    positions: Sequence[int],
) -> tuple[EvaluationWindow, list[list[int]]]:
    source_texts = [src for src, _ in pairs]
    target_texts = [tgt for _, tgt in pairs]
    src_ids, src_masks, src_positions = _tokenize_texts_padded(
        source_texts,
        tokenizer,
        seq_len,
        positions=positions,
    )
    tgt_ids, tgt_masks, tgt_positions = _tokenize_texts_padded(
        target_texts,
        tokenizer,
        seq_len,
        positions=positions,
    )
    src_map = {
        position: (input_ids, attention_mask)
        for position, input_ids, attention_mask in zip(
            src_positions, src_ids, src_masks, strict=False
        )
    }
    tgt_map = {
        position: [
            int(token) if int(mask) else -100
            for token, mask in zip(target_ids, target_mask, strict=False)
        ]
        for position, target_ids, target_mask in zip(
            tgt_positions, tgt_ids, tgt_masks, strict=False
        )
    }
    kept_positions = [
        int(position)
        for position in positions
        if position in src_map and position in tgt_map
    ]
    window = EvaluationWindow(
        [src_map[position][0] for position in kept_positions],
        [src_map[position][1] for position in kept_positions],
        kept_positions,
    )
    labels = [tgt_map[position] for position in kept_positions]
    return window, labels


def _split_labels_by_index(
    labels: Sequence[list[int]],
    indices: Sequence[int],
    *,
    split_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    preview_labels: list[list[int]] = []
    final_labels: list[list[int]] = []
    for index, label in zip(indices, labels, strict=False):
        if index < split_index:
            preview_labels.append(label)
        else:
            final_labels.append(label)
    return preview_labels, final_labels


def _local_files_signature(files: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for file_path in files:
        try:
            stat = file_path.stat()
            signature.append(
                (file_path.as_posix(), int(stat.st_mtime_ns), int(stat.st_size))
            )
        except OSError:
            signature.append((file_path.as_posix(), -1, -1))
    return tuple(signature)


class DatasetProvider(Protocol):
    """
    Protocol for pluggable dataset providers.

    Enables extensible dataset support while maintaining deterministic evaluation.
    """

    name: str

    @abstractmethod
    def load(self, split: str = "validation", **kwargs) -> list[str]:
        """
        Load raw text samples from the dataset.

        Args:
            split: Dataset split to load ("validation", "test", "train")
            **kwargs: Provider-specific parameters

        Returns:
            List of text strings
        """
        ...

    @abstractmethod
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
        """
        Create deterministic preview and final evaluation windows.

        Args:
            tokenizer: Tokenizer to use for text encoding
            seq_len: Maximum sequence length
            stride: Stride for overlapping windows (unused in current impl)
            preview_n: Number of preview samples
            final_n: Number of final samples
            seed: Random seed for deterministic sampling
            split: Dataset split to use

        Returns:
            Tuple of (preview_window, final_window)
        """
        ...

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
        """
        Estimate number of non-overlapping, deduplicated windows available for evaluation.

        Returns metadata describing the available capacity (total tokens, usable windows, dedupe rate).
        """
        ...

    def info(self) -> dict[str, Any]:
        """Get information about this dataset provider."""
        return {"name": self.name, "type": "dataset_provider"}


class WikiText2Provider:
    """
    WikiText-2 dataset provider with deterministic windowing.

    Implements the canonical WT-2 evaluation setup with fixed 100+100 preview/final samples.
    """

    name = "wikitext2"
    _BYTE_NGRAM_ORDER = 4
    _BYTE_NGRAM_PAD = 256
    _BYTE_NGRAM_ALPHA = 1.0

    def __init__(
        self,
        cache_dir: Path | None = None,
        device_hint: str | None = None,
        emit: EventEmitter | None = None,
        **_: Any,
    ):
        """
        Initialize WikiText-2 provider.

        Args:
            cache_dir: Optional cache directory for dataset storage
        """
        self.cache_dir = cache_dir
        self._emit_event = emit
        self._validate_dependencies()
        self._last_stratification_stats: dict[str, Any] | None = None
        self._last_batch_size_used: int = 0
        self._last_scorer_profile: dict[str, Any] | None = None
        # In-process cache for loaded/filtered texts to avoid repeated
        # load_dataset() calls across stratification retries.
        self._texts_cache: dict[str, list[str]] = {}
        # Optional device hint from CLI/resolved run device (e.g. "cpu", "cuda", "mps", "auto")
        normalized_hint = (device_hint or "").strip().lower()
        self._device_hint: str | None = normalized_hint or None

    def _event(self, tag: str, message: str, *, emoji: str | None = None) -> None:
        """Emit a dataset event via an optional CLI-provided sink."""
        if self._emit_event is None:
            return
        self._emit_event(tag, message, emoji)

    def _validate_dependencies(self) -> None:
        """Check that required dependencies are available."""
        _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for WikiText-2 loading"
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
        """Estimate available non-overlapping windows for evaluation."""
        texts = self.load(split=split, max_samples=2000)
        if not texts:
            return {
                "total_tokens": 0,
                "available_nonoverlap": 0,
                "available_unique": 0,
                "dedupe_rate": 0.0,
                "stride": stride,
                "seq_len": seq_len,
                "candidate_unique": 0,
                "candidate_limit": 0,
            }

        env_fast = os.environ.get("INVARLOCK_CAPACITY_FAST", "")
        env_fast_flag = isinstance(env_fast, str) and env_fast.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        use_fast = bool(fast_mode) or env_fast_flag
        if use_fast:
            base_available = len(texts)
            target_total = int(target_total or 0)
            approx_available = base_available
            if target_total > 0:
                approx_available = max(base_available, target_total)
            total_tokens = int(max(approx_available, 0) * seq_len)
            approx_available = int(max(approx_available, 0))
            return {
                "total_tokens": total_tokens,
                "available_nonoverlap": approx_available,
                "available_unique": approx_available,
                "dedupe_rate": 0.0,
                "stride": int(stride),
                "seq_len": int(seq_len),
                "candidate_unique": approx_available,
                "candidate_limit": approx_available,
            }

        tokenized = self._collect_tokenized_samples(
            texts, list(range(len(texts))), tokenizer, seq_len
        )

        total_tokens = sum(item[3] for item in tokenized)
        available_nonoverlap = len(tokenized)

        unique_sequences: set[tuple[int, ...]] = set()
        for _, input_ids, attention_mask, _ in tokenized:
            seq = tuple(
                int(tok_id)
                for tok_id, mask in zip(input_ids, attention_mask, strict=False)
                if mask
            )
            unique_sequences.add(seq)

        available_unique = len(unique_sequences)
        dedupe_rate = (
            0.0
            if available_nonoverlap == 0
            else max(
                0.0,
                1.0 - (available_unique / float(max(available_nonoverlap, 1))),
            )
        )

        candidate_unique = None
        candidate_limit = None
        if target_total is not None and target_total > 0:
            reserve_buffer = max(int(target_total * 0.2), 64)
            candidate_limit = min(len(texts), target_total + reserve_buffer)
            tokenized_subset = self._collect_tokenized_samples(
                texts, list(range(candidate_limit)), tokenizer, seq_len
            )
            subset_signatures = {
                tuple(
                    int(tok)
                    for tok, mask in zip(entry[1], entry[2], strict=False)
                    if mask
                )
                for entry in tokenized_subset
            }
            candidate_unique = len(subset_signatures)

        result = {
            "total_tokens": int(total_tokens),
            "available_nonoverlap": int(available_nonoverlap),
            "available_unique": int(available_unique),
            "dedupe_rate": float(dedupe_rate),
            "stride": int(stride),
            "seq_len": int(seq_len),
        }
        if candidate_unique is not None:
            result["candidate_unique"] = int(candidate_unique)
            result["candidate_limit"] = int(candidate_limit or 0)
        return result

    def load(
        self, split: str = "validation", max_samples: int = 2000, **kwargs
    ) -> list[str]:
        """
        Load WikiText-2 text samples.

        Args:
            split: Dataset split ("validation", "test", "train")
            max_samples: Maximum samples to load
            **kwargs: Additional parameters (ignored)

        Returns:
            List of filtered text strings
        """
        self._event(
            "DATA",
            f"WikiText-2 {split}: loading split...",
            emoji="📚",
        )

        # Serve from cache when possible (load the largest slice once)
        cached = self._texts_cache.get(split)
        if cached is not None and len(cached) >= max_samples:
            return cached[:max_samples]

        # Load dataset with size limit for efficiency
        dataset_slice = f"{split}[:{max_samples}]" if max_samples > 0 else split
        load_dataset = _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for WikiText-2 loading"
        )
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=dataset_slice,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )

        # Filter out empty/short texts
        valid_texts: list[str] = []
        for item in dataset:
            text = str(item.get("text", "")).strip()
            # Keep texts with at least 20 characters and some alphabetic content
            if len(text) >= 20 and any(c.isalpha() for c in text):
                valid_texts.append(text)

        # Optional exact-text dedupe to reduce duplicate-token windows
        # Enable via INVARLOCK_DEDUP_TEXTS=1 (keeps first occurrence, preserves order)
        import os as _os

        if str(_os.environ.get("INVARLOCK_DEDUP_TEXTS", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            seen: set[str] = set()
            deduped: list[str] = []
            for t in valid_texts:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)
            valid_texts = deduped

        # Cache the largest slice we’ve seen for this split
        prev = self._texts_cache.get(split)
        if prev is None or len(valid_texts) > len(prev):
            self._texts_cache[split] = list(valid_texts)

        self._event(
            "DATA",
            f"Loaded {len(valid_texts)}/{len(dataset)} valid samples",
        )
        return valid_texts

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
        """
        Create deterministic preview and final evaluation windows.

        This implements the core deterministic evaluation requirement:
        - Fixed seed ensures reproducible sample selection
        - Non-overlapping preview and final samples
        - Consistent tokenization parameters

        Args:
            tokenizer: HuggingFace tokenizer for text encoding
            seq_len: Maximum sequence length for tokenization
            stride: Stride parameter (reserved for future use)
            preview_n: Number of preview samples (default: 100)
            final_n: Number of final samples (default: 100)
            seed: Random seed for reproducible sampling
            split: Dataset split to use

        Returns:
            Tuple of (preview_window, final_window) with deterministic samples
        """
        total_required = preview_n + final_n
        if total_required <= 0:
            raise _ValErr(
                code="E302", message="VALIDATION-FAILED: preview/final must be positive"
            )

        # Load text data with additional buffer to ensure enough valid samples for release windows.
        extra_pool = max(500, int(0.5 * total_required))
        max_samples = max(total_required + extra_pool, 2000)
        texts = self.load(split=split, max_samples=max_samples)

        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(len(texts)).tolist()

        reserve = max(16, int(0.1 * total_required))
        target_pool = min(len(texts), total_required + reserve * 2)

        if target_pool < total_required:
            raise _DataErr(
                code="E303",
                message=(
                    "CAPACITY-INSUFFICIENT: not enough valid samples for requested preview/final"
                ),
                details={
                    "have": int(len(texts)),
                    "preview": int(preview_n),
                    "final": int(final_n),
                },
            )

        candidates: list[dict[str, Any]] = []
        used_indices: set[int] = set()
        cursor = 0
        chunk_size = max(64, min(256, target_pool))

        self._event(
            "DATA",
            "Creating evaluation windows:",
            emoji="📊",
        )
        self._event("DATA", f"Requested preview/final: {preview_n}/{final_n}")
        self._event("DATA", f"Sampling pool target: {target_pool} (reserve {reserve})")

        while len(candidates) < total_required + reserve and cursor < len(
            shuffled_indices
        ):
            batch = shuffled_indices[cursor : cursor + chunk_size]
            cursor += chunk_size

            tokenized_batch = self._collect_tokenized_samples(
                texts, batch, tokenizer, seq_len
            )

            for (
                idx,
                input_ids_list,
                attention_mask_list,
                real_tokens,
            ) in tokenized_batch:
                if idx in used_indices:
                    continue
                used_indices.add(idx)
                candidates.append(
                    {
                        "dataset_index": idx,
                        "text": texts[idx],
                        "input_ids": input_ids_list,
                        "attention_mask": attention_mask_list,
                        "token_count": real_tokens,
                        "seq_len": len(input_ids_list),
                    }
                )

            if cursor >= len(shuffled_indices) and len(candidates) < total_required:
                break

        if len(candidates) < total_required:
            raise _DataErr(
                code="E304",
                message=(
                    "TOKENIZE-INSUFFICIENT: failed to gather enough tokenized samples"
                ),
                details={"needed": int(total_required), "got": int(len(candidates))},
            )

        self._score_candidates_byte_ngram(candidates)

        sorted_candidates = sorted(
            candidates, key=lambda item: (item["difficulty"], item["dataset_index"])
        )

        total_candidates = len(sorted_candidates)
        selection_count = total_required
        selected_positions: list[int] = []
        used_positions: set[int] = set()

        for k in range(selection_count):
            target_position = (k + 0.5) * total_candidates / selection_count
            base_idx = int(round(target_position))
            offset = 0
            chosen: int | None = None

            while offset < total_candidates:
                for candidate_idx in (base_idx + offset, base_idx - offset):
                    if (
                        0 <= candidate_idx < total_candidates
                        and candidate_idx not in used_positions
                    ):
                        chosen = candidate_idx
                        break
                if chosen is not None:
                    break
                offset += 1

            if chosen is not None:
                used_positions.add(chosen)
                selected_positions.append(chosen)

        if len(selected_positions) < selection_count:
            for candidate_idx in range(total_candidates):
                if candidate_idx not in used_positions:
                    used_positions.add(candidate_idx)
                    selected_positions.append(candidate_idx)
                if len(selected_positions) == selection_count:
                    break

        if len(selected_positions) < selection_count:
            raise _DataErr(
                code="E305", message="STRATIFY-FAILED: candidate pool insufficient"
            )

        selected_candidates = [sorted_candidates[idx] for idx in selected_positions]
        selected_candidates.sort(
            key=lambda item: (item["difficulty"], item["dataset_index"])
        )

        preview_candidates: list[dict[str, Any]] = []
        final_candidates: list[dict[str, Any]] = []

        def assign_candidate(
            candidate: dict[str, Any],
            primary: list[dict[str, Any]],
            secondary: list[dict[str, Any]],
            primary_capacity: int,
            secondary_capacity: int,
        ) -> None:
            if len(primary) < primary_capacity:
                primary.append(candidate)
            elif len(secondary) < secondary_capacity:
                secondary.append(candidate)

        for pair_start in range(0, len(selected_candidates), 2):
            pair = selected_candidates[pair_start : pair_start + 2]
            if not pair:
                continue
            if len(pair) == 2:
                easy, hard = pair
                pair_index = pair_start // 2
                if pair_index % 2 == 0:
                    assign_candidate(
                        easy, preview_candidates, final_candidates, preview_n, final_n
                    )
                    assign_candidate(
                        hard, final_candidates, preview_candidates, final_n, preview_n
                    )
                else:
                    assign_candidate(
                        easy, final_candidates, preview_candidates, final_n, preview_n
                    )
                    assign_candidate(
                        hard, preview_candidates, final_candidates, preview_n, final_n
                    )
            else:
                lone_candidate = pair[0]
                assign_candidate(
                    lone_candidate,
                    preview_candidates,
                    final_candidates,
                    preview_n,
                    final_n,
                )

        assigned_ids = {
            id(candidate) for candidate in preview_candidates + final_candidates
        }
        remaining = [
            candidate
            for candidate in selected_candidates
            if id(candidate) not in assigned_ids
        ]
        for candidate in remaining:
            if len(preview_candidates) < preview_n:
                preview_candidates.append(candidate)
            elif len(final_candidates) < final_n:
                final_candidates.append(candidate)

        def _mean_difficulty(candidates: list[dict[str, Any]]) -> float:
            if not candidates:
                return 0.0
            return float(
                sum(candidate["difficulty"] for candidate in candidates)
                / len(candidates)
            )

        for _ in range(100):
            if not preview_candidates or not final_candidates:
                break
            diff = _mean_difficulty(preview_candidates) - _mean_difficulty(
                final_candidates
            )
            if abs(diff) <= 1e-4:
                break
            if diff < 0:
                preview_candidate = min(
                    preview_candidates, key=lambda c: c["difficulty"]
                )
                final_candidate = max(final_candidates, key=lambda c: c["difficulty"])
            else:
                preview_candidate = max(
                    preview_candidates, key=lambda c: c["difficulty"]
                )
                final_candidate = min(final_candidates, key=lambda c: c["difficulty"])

            if preview_candidate is final_candidate:
                break

            preview_candidates.remove(preview_candidate)
            final_candidates.remove(final_candidate)
            preview_candidates.append(final_candidate)
            final_candidates.append(preview_candidate)

            new_diff = _mean_difficulty(preview_candidates) - _mean_difficulty(
                final_candidates
            )
            if abs(new_diff) >= abs(diff) - 1e-6:
                # swap did not improve; revert and stop
                preview_candidates.remove(final_candidate)
                final_candidates.remove(preview_candidate)
                preview_candidates.append(preview_candidate)
                final_candidates.append(final_candidate)
                break

        if len(preview_candidates) != preview_n or len(final_candidates) != final_n:
            raise _DataErr(
                code="E305",
                message=(
                    "STRATIFY-FAILED: failed to allocate preview/final windows with equal counts"
                ),
                details={
                    "preview_target": int(preview_n),
                    "final_target": int(final_n),
                    "preview_got": int(len(preview_candidates)),
                    "final_got": int(len(final_candidates)),
                },
            )

        preview_candidates.sort(
            key=lambda item: (item["difficulty"], item["dataset_index"])
        )
        final_candidates.sort(
            key=lambda item: (item["difficulty"], item["dataset_index"])
        )

        preview_window = EvaluationWindow(
            input_ids=[c["input_ids"] for c in preview_candidates],
            attention_masks=[c["attention_mask"] for c in preview_candidates],
            indices=[c["dataset_index"] for c in preview_candidates],
        )

        final_window = EvaluationWindow(
            input_ids=[c["input_ids"] for c in final_candidates],
            attention_masks=[c["attention_mask"] for c in final_candidates],
            indices=[c["dataset_index"] for c in final_candidates],
        )

        if len(preview_window) != preview_n or len(final_window) != final_n:
            raise _DataErr(
                code="E305",
                message="STRATIFY-FAILED: window stratification mismatch",
                details={
                    "preview_target": int(preview_n),
                    "final_target": int(final_n),
                    "preview_got": int(len(preview_window)),
                    "final_got": int(len(final_window)),
                },
            )

        preview_difficulties = [c["difficulty"] for c in preview_candidates]
        final_difficulties = [c["difficulty"] for c in final_candidates]
        self._last_stratification_stats = {
            "pool_size": len(selected_candidates),
            "reserve": reserve,
            "batch_size_used": int(self._last_batch_size_used),
            "preview_mean_difficulty": float(np.mean(preview_difficulties))
            if preview_difficulties
            else 0.0,
            "final_mean_difficulty": float(np.mean(final_difficulties))
            if final_difficulties
            else 0.0,
            "preview_std_difficulty": float(np.std(preview_difficulties))
            if preview_difficulties
            else 0.0,
            "final_std_difficulty": float(np.std(final_difficulties))
            if final_difficulties
            else 0.0,
            "difficulty_gap": float(
                (np.mean(final_difficulties) - np.mean(preview_difficulties))
                if (preview_difficulties and final_difficulties)
                else 0.0
            ),
        }

        self._event("DATA", f"Seed: {seed}, Seq length: {seq_len}")
        self._event("DATA", f"Preview: {len(preview_window)} samples")
        self._event("DATA", f"Final: {len(final_window)} samples")

        return preview_window, final_window

    def _collect_tokenized_samples(
        self,
        texts: Sequence[str],
        indices: Sequence[int],
        tokenizer: Any,
        seq_len: int,
    ) -> list[tuple[int, list[int], list[int], int]]:
        """Tokenize samples and return raw sequences without logging."""
        batch_texts: list[str] = []
        batch_indices: list[int] = []
        for idx in indices:
            if idx >= len(texts):
                continue
            batch_indices.append(int(idx))
            batch_texts.append(texts[idx])

        input_ids_list, attention_masks_list, valid_indices = _tokenize_texts_padded(
            batch_texts,
            tokenizer,
            seq_len,
            positions=batch_indices,
            warn_on_failure=True,
        )

        results: list[tuple[int, list[int], list[int], int]] = []
        for idx, input_ids, attention_mask in zip(
            valid_indices,
            input_ids_list,
            attention_masks_list,
            strict=False,
        ):
            real_tokens = int(sum(attention_mask))
            if real_tokens > 1:
                results.append((idx, input_ids, attention_mask, real_tokens))

        return results

    def _score_candidates_byte_ngram(self, candidates: list[dict[str, Any]]) -> bool:
        if not candidates:
            self._last_batch_size_used = 0
            self._last_scorer_profile = None
            return False

        order = max(1, int(self._BYTE_NGRAM_ORDER))
        pad_token = int(self._BYTE_NGRAM_PAD)
        alpha = float(self._BYTE_NGRAM_ALPHA)
        vocab_size = pad_token + 1
        context_width = max(order - 1, 0)
        context_modulus = vocab_size ** max(context_width - 1, 0)

        def _initial_context_key() -> int:
            key = 0
            for _ in range(context_width):
                key = (key * vocab_size) + pad_token
            return key

        def _next_context_key(current_key: int, token: int) -> int:
            if context_width <= 0:
                return 0
            if context_width == 1:
                return int(token)
            return int((current_key % context_modulus) * vocab_size + int(token))

        context_counts: Counter[int] = Counter()
        ngram_counts: Counter[int] = Counter()
        sequences: list[bytes] = []
        start_time = time.perf_counter()

        for candidate in candidates:
            text = candidate.get("text")
            if not isinstance(text, str):
                text = ""
            byte_values = text.encode("utf-8", errors="replace")
            sequences.append(byte_values)
            context_key = _initial_context_key()
            for token in byte_values:
                context_counts[context_key] += 1
                ngram_counts[(context_key * vocab_size) + int(token)] += 1
                context_key = _next_context_key(context_key, int(token))

        total_tokens = 0
        for candidate, byte_values in zip(candidates, sequences, strict=False):
            loss_sum = 0.0
            token_count = 0
            context_key = _initial_context_key()
            for token in byte_values:
                context_count = context_counts.get(context_key, 0)
                ngram_key = (context_key * vocab_size) + int(token)
                ngram_count = ngram_counts.get(ngram_key, 0)
                prob = (ngram_count + alpha) / (context_count + alpha * vocab_size)
                loss_sum += -math.log(prob)
                token_count += 1
                context_key = _next_context_key(context_key, int(token))
            candidate["difficulty"] = loss_sum / max(token_count, 1)
            total_tokens += token_count

        self._last_batch_size_used = len(candidates)
        elapsed = max(time.perf_counter() - start_time, 1e-9)
        tokens_per_sec = total_tokens / elapsed if total_tokens else 0.0
        self._last_scorer_profile = {
            "mode": "byte_ngram",
            "order": order,
            "vocab_size": vocab_size,
            "tokens_processed": total_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_per_sec,
        }
        return True

    def _tokenize_samples(
        self,
        texts: list[str],
        indices: list[int],
        tokenizer: Any,
        seq_len: int,
        window_name: str,
    ) -> EvaluationWindow:
        """Tokenize a set of text samples with consistent parameters."""
        collected = self._collect_tokenized_samples(texts, indices, tokenizer, seq_len)

        input_ids_list = [entry[1] for entry in collected]
        attention_masks_list = [entry[2] for entry in collected]
        valid_indices = [entry[0] for entry in collected]

        self._event(
            "DATA",
            f"{window_name}: {len(valid_indices)}/{len(indices)} samples tokenized",
        )

        return EvaluationWindow(
            input_ids=input_ids_list,
            attention_masks=attention_masks_list,
            indices=valid_indices,
        )

    @property
    def stratification_stats(self) -> dict[str, Any] | None:
        """Return summary statistics for the most recent stratified split."""
        return self._last_stratification_stats

    @property
    def scorer_profile(self) -> dict[str, Any] | None:
        """Return performance statistics for the most recent scorer run."""
        return self._last_scorer_profile

    def info(self) -> dict[str, Any]:
        """Get information about WikiText-2 provider."""
        return {
            "name": self.name,
            "type": "dataset_provider",
            "dataset": "wikitext-2-raw-v1",
            "source": "huggingface/datasets",
            "deterministic": True,
            "default_split": "validation",
            "requires": ["datasets"],
        }


class SyntheticProvider:
    """
    Synthetic text provider for testing and development.

    Generates coherent text samples when WikiText-2 is not available.
    """

    name = "synthetic"

    def __init__(self, base_samples: list[str] | None = None):
        """Initialize with optional base text samples."""
        self.base_samples = base_samples or self._default_samples()
        self._load_cache: dict[int, list[str]] = {}

    def _default_samples(self) -> list[str]:
        """Generate default synthetic text samples."""
        return [
            "The weather today is quite pleasant with clear skies and gentle winds.",
            "Scientists have discovered a new species in the Amazon rainforest region.",
            "The stock market showed significant gains during this quarter's trading.",
            "Technology companies are investing heavily in artificial intelligence research.",
            "The new restaurant downtown serves excellent Mediterranean cuisine daily.",
            "Climate change continues to affect global weather patterns significantly.",
            "The university announced new programs in data science and engineering.",
            "Renewable energy sources are becoming more cost-effective than fossil fuels.",
            "The museum exhibition features artwork from the Renaissance period.",
            "Public transportation systems are being upgraded in major cities worldwide.",
            "Medical researchers published breakthrough findings about genetic therapy.",
            "The concert hall will host a performance by the symphony orchestra.",
            "Local farmers are adopting sustainable agricultural practices this season.",
            "The new software update includes enhanced security features and performance.",
            "International trade agreements are being renegotiated between countries.",
        ]

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
        """Synthetic provider offers deterministic capacity based on base samples."""
        total_tokens = len(self.base_samples) * seq_len
        available = len(self.base_samples)
        return {
            "total_tokens": int(total_tokens),
            "available_nonoverlap": int(available),
            "available_unique": int(available),
            "dedupe_rate": 0.0,
            "stride": int(stride),
            "seq_len": int(seq_len),
            "candidate_unique": int(available),
            "candidate_limit": int(available),
        }

    def load(
        self, split: str = "validation", max_samples: int = 500, **kwargs
    ) -> list[str]:
        """Generate synthetic text samples."""
        cached = self._load_cache.get(int(max_samples))
        if cached is not None:
            return cached

        # Expand base samples to meet requirement, preferring unique variations
        # to avoid duplicate-token windows (important for stratified pairing).
        expanded_samples: list[str] = []
        variations = [
            lambda s: s,
            lambda s: f"Recently, {s.lower()}",
            lambda s: f"According to reports, {s.lower()}",
            lambda s: f"It is notable that {s.lower()}",
            lambda s: f"Furthermore, {s.lower()}",
            lambda s: f"In addition, {s.lower()}",
        ]
        # Deterministic coverage of (variation × base sample) combinations first.
        for variation in variations:
            for base_text in self.base_samples:
                expanded_samples.append(variation(base_text))
                if len(expanded_samples) >= max_samples:
                    self._load_cache[int(max_samples)] = expanded_samples
                    return expanded_samples

        # If callers request more than the unique combination space, keep
        # extending deterministically while ensuring uniqueness via a suffix.
        idx = 0
        while len(expanded_samples) < max_samples:
            base_text = self.base_samples[idx % len(self.base_samples)]
            variation = variations[(idx // len(self.base_samples)) % len(variations)]
            expanded_samples.append(
                f"{variation(base_text)} [synthetic #{len(expanded_samples)}]"
            )
            idx += 1

        self._load_cache[int(max_samples)] = expanded_samples
        return expanded_samples

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
        """Create synthetic evaluation windows."""
        texts = self.load(split=split, max_samples=preview_n + final_n)
        total = min(len(texts), int(preview_n) + int(final_n))
        combined_window = self._simple_tokenize(
            texts[:total], tokenizer, seq_len, list(range(total))
        )
        return _split_window_by_index(combined_window, split_index=preview_n)

    def _simple_tokenize(
        self, texts: list[str], tokenizer: Any, seq_len: int, indices: list[int]
    ) -> EvaluationWindow:
        """Simple tokenization for synthetic samples."""
        if callable(tokenizer) or hasattr(tokenizer, "encode"):
            input_ids_list, attention_masks_list, valid_indices = (
                _tokenize_texts_padded(
                    texts,
                    tokenizer,
                    seq_len,
                    positions=indices,
                )
            )
            if input_ids_list:
                return EvaluationWindow(
                    input_ids=input_ids_list,
                    attention_masks=attention_masks_list,
                    indices=valid_indices,
                )

        input_ids_list: list[list[int]] = []
        attention_masks_list: list[list[int]] = []
        for _ in texts:
            # Fallback for lightweight test scenarios without tokenizer support.
            input_ids = list(range(1, min(seq_len + 1, 50))) + [0] * max(
                0, seq_len - 49
            )
            attention_mask = [1] * min(seq_len, 49) + [0] * max(0, seq_len - 49)
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)

        return EvaluationWindow(
            input_ids=input_ids_list,
            attention_masks=attention_masks_list,
            indices=list(indices[: len(input_ids_list)]),
        )

    def info(self) -> dict[str, Any]:
        """Get information about synthetic provider."""
        return {
            "name": self.name,
            "type": "dataset_provider",
            "dataset": "synthetic",
            "source": "generated",
            "deterministic": True,
            "base_samples": len(self.base_samples),
        }


class HFTextProvider:
    """
    Generic HuggingFace datasets text provider.

    Loads a text dataset by name/config and extracts a specified text field.
    Provides simple deterministic windowing suitable for CI/demo usage.
    """

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
        _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for hf_text provider"
        )
        self.dataset_name = dataset_name or "wikitext"
        self.config_name = config_name or None
        self.text_field = text_field
        self.cache_dir = cache_dir
        self.trust_remote_code = bool(trust_remote_code)
        self.max_samples = int(max_samples)
        self._texts_cache: dict[str, list[str]] = {}

    def load(self, split: str = "validation", **kwargs) -> list[str]:
        cached = self._texts_cache.get(split)
        if cached is not None:
            return cached
        load_dataset = _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for hf_text provider"
        )
        ds = load_dataset(
            path=self.dataset_name,
            name=self.config_name,
            split=split,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        texts: list[str] = []
        # Limit to max_samples for CI friendliness
        count = 0
        for row in ds:
            if self.text_field not in row:
                continue
            val = row[self.text_field]
            if isinstance(val, str) and val.strip():
                texts.append(val)
                count += 1
                if count >= self.max_samples:
                    break
        self._texts_cache[split] = texts
        return texts

    def _simple_tokenize(
        self, texts: list[str], tokenizer: Any, seq_len: int, indices: list[int]
    ) -> EvaluationWindow:
        input_ids_list, attention_masks_list, valid_indices = _tokenize_texts_padded(
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
        total = len(texts)
        if total == 0:
            # Typed-only: no-samples is a DataError for consistency
            raise _DataErr(
                code="E306",
                message=(
                    "NO-SAMPLES: hf_text produced no samples; check dataset_name/config_name/text_field"
                ),
            )
        total_required = int(preview_n) + int(final_n)
        if total_required <= 0:
            raise _ValErr(
                code="E302", message="VALIDATION-FAILED: preview/final must be positive"
            )
        # Deterministic seeded sampling: avoid contiguous split bias on long
        # training corpora where early records can differ materially from later
        # ones. Continue sampling until enough unique non-trivial token windows
        # are collected so CI lanes do not collapse on padded/duplicate rows.
        selected_positions = list(range(len(texts)))
        random.Random(int(seed)).shuffle(selected_positions)

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
        preview_window = EvaluationWindow(
            input_ids=[sample[1] for sample in preview_samples],
            attention_masks=[sample[2] for sample in preview_samples],
            indices=[sample[0] for sample in preview_samples],
        )
        final_window = EvaluationWindow(
            input_ids=[sample[1] for sample in final_samples],
            attention_masks=[sample[2] for sample in final_samples],
            indices=[sample[0] for sample in final_samples],
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
    """HuggingFace seq2seq provider with paired source/target fields.

    Loads a dataset with text pairs and exposes encoder input_ids/attention_masks.
    Decoder target token ids are exposed via last_preview_labels / last_final_labels
    for the runner to attach as labels.
    """

    name = "hf_seq2seq"

    def __init__(
        self,
        dataset_name: str,
        config_name: str | None = None,
        src_field: str = "source",
        tgt_field: str = "target",
        cache_dir: str | None = None,
        max_samples: int = 2000,
    ) -> None:
        _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for hf_seq2seq provider"
        )
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.cache_dir = cache_dir
        self.max_samples = int(max_samples)
        self.last_preview_labels: list[list[int]] | None = None
        self.last_final_labels: list[list[int]] | None = None
        self._pairs_cache: dict[str, list[tuple[str, str]]] = {}

    def _load_pairs(self, split: str) -> list[tuple[str, str]]:
        cached = self._pairs_cache.get(split)
        if cached is not None:
            return cached
        load_dataset = _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for hf_seq2seq provider"
        )
        ds = load_dataset(
            path=self.dataset_name,
            name=self.config_name,
            split=split,
            cache_dir=self.cache_dir,
        )
        out: list[tuple[str, str]] = []
        count = 0
        for row in ds:
            src = row.get(self.src_field)
            tgt = row.get(self.tgt_field)
            if (
                isinstance(src, str)
                and src.strip()
                and isinstance(tgt, str)
                and tgt.strip()
            ):
                out.append((src, tgt))
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
        # Deterministic slicing
        prev_pairs = pairs[:preview_n]
        fin_pairs = pairs[preview_n : preview_n + final_n]
        combined_pairs = prev_pairs + fin_pairs
        combined_positions = list(range(len(prev_pairs))) + list(
            range(preview_n, preview_n + len(fin_pairs))
        )
        combined_window, combined_labels = _tokenize_combined_pairs(
            combined_pairs,
            tokenizer=tokenizer,
            seq_len=seq_len,
            positions=combined_positions,
        )
        preview_window, final_window = _split_window_by_index(
            combined_window, split_index=preview_n
        )
        self.last_preview_labels, self.last_final_labels = _split_labels_by_index(
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


class LocalJSONLProvider:
    """
    Local JSONL provider for BYOD text datasets.

    Accepts a single `file`, a `path` (file or directory), or `data_files`
    (glob or list of paths). Extracts a `text_field` (defaults to "text").
    """

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
        files: list[Path] = []
        # Explicit file
        if isinstance(self.file, str) and self.file:
            p = Path(self.file)
            if p.exists() and p.is_file():
                files.append(p)
        # Path can be file or directory
        if isinstance(self.path, str) and self.path:
            p = Path(self.path)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                files.extend(sorted(p.glob("*.jsonl")))
        # data_files may be a glob or list
        if isinstance(self.data_files, str) and self.data_files:
            from glob import glob as _glob

            files.extend(Path(p) for p in _glob(self.data_files))
        elif isinstance(self.data_files, list):
            for item in self.data_files:
                try:
                    pp = Path(str(item))
                    if pp.exists() and pp.is_file():
                        files.append(pp)
                except Exception:
                    continue
        # Deduplicate while preserving order
        seen: set[str] = set()
        uniq: list[Path] = []
        for f in files:
            fp = f.resolve().as_posix()
            if fp not in seen:
                seen.add(fp)
                uniq.append(f)
        return uniq

    def load(self, split: str = "validation", **kwargs) -> list[str]:
        files = self._resolve_files()
        cache_key = (_local_files_signature(files), self.text_field, self.max_samples)
        if self._load_cache is not None and self._load_cache[0] == cache_key:
            return self._load_cache[1]
        texts: list[str] = []
        count = 0
        for fp in files:
            try:
                with fp.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        val = obj.get(self.text_field)
                        if isinstance(val, str) and val.strip():
                            texts.append(val)
                            count += 1
                            if count >= self.max_samples:
                                self._load_cache = (cache_key, texts)
                                return texts
            except Exception:
                continue
        self._load_cache = (cache_key, texts)
        return texts

    def _simple_tokenize(
        self, texts: list[str], tokenizer: Any, seq_len: int, indices: list[int]
    ) -> EvaluationWindow:
        input_ids_list, attention_masks_list, valid_indices = _tokenize_texts_padded(
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
        return _split_window_by_index(combined_window, split_index=preview_n)

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
    """Local JSONL pairs provider with source/target fields.

    Accepts a single `file`, a `path` (file or directory), or `data_files`
    (glob or list of paths). Extracts paired strings from `src_field`/`tgt_field`.
    """

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
        files: list[Path] = []
        if isinstance(self.file, str) and self.file:
            p = Path(self.file)
            if p.exists() and p.is_file():
                files.append(p)
        if isinstance(self.path, str) and self.path:
            p = Path(self.path)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                files.extend(sorted(p.glob("*.jsonl")))
        if isinstance(self.data_files, str) and self.data_files:
            from glob import glob as _glob

            files.extend(Path(p) for p in _glob(self.data_files))
        elif isinstance(self.data_files, list):
            for item in self.data_files:
                try:
                    pp = Path(str(item))
                    if pp.exists() and pp.is_file():
                        files.append(pp)
                except Exception:
                    continue
        # Deduplicate
        seen: set[str] = set()
        uniq: list[Path] = []
        for f in files:
            fp = f.resolve().as_posix()
            if fp not in seen:
                seen.add(fp)
                uniq.append(f)
        return uniq

    def _load_pairs(self) -> list[tuple[str, str]]:
        files = self._resolve_files()
        cache_key = (
            _local_files_signature(files),
            self.src_field,
            self.tgt_field,
            self.max_samples,
        )
        if self._pairs_cache is not None and self._pairs_cache[0] == cache_key:
            return self._pairs_cache[1]
        pairs: list[tuple[str, str]] = []
        count = 0
        for fp in files:
            try:
                with fp.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        src = obj.get(self.src_field)
                        tgt = obj.get(self.tgt_field)
                        if (
                            isinstance(src, str)
                            and src.strip()
                            and isinstance(tgt, str)
                            and tgt.strip()
                        ):
                            pairs.append((src, tgt))
                            count += 1
                            if count >= self.max_samples:
                                self._pairs_cache = (cache_key, pairs)
                                return pairs
            except Exception:
                continue
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
        combined_window, combined_labels = _tokenize_combined_pairs(
            combined_pairs,
            tokenizer=tokenizer,
            seq_len=seq_len,
            positions=combined_positions,
        )
        preview_window, final_window = _split_window_by_index(
            combined_window, split_index=preview_n
        )
        self.last_preview_labels, self.last_final_labels = _split_labels_by_index(
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

    # (text-only helpers removed; LocalJSONLProvider implements text tokenization)


class Seq2SeqDataProvider:
    """Synthetic seq2seq provider wrapper to fit DatasetProvider interface.

    Bridges invarlock.eval.providers.seq2seq.Seq2SeqProvider to the windowing
    protocol used by the CLI runner. Generates encoder input_ids from src_ids,
    attention_masks from src_mask, and allows the runner to derive labels.
    """

    name = "seq2seq"

    def __init__(self, **kwargs: Any) -> None:
        # Pass through kwargs to underlying provider (n, src_len, tgt_len, pad_id, bos_id, eos_id)
        from invarlock.eval.providers.seq2seq import Seq2SeqProvider as _S2S

        self._inner = _S2S(**kwargs)
        self.last_preview_labels: list[list[int]] | None = None
        self.last_final_labels: list[list[int]] | None = None

    def load(
        self, split: str = "validation", **kwargs
    ) -> list[str]:  # pragma: no cover - not used
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
        # Generate exactly preview_n + final_n examples deterministically
        total = max(0, int(preview_n) + int(final_n))
        if total <= 0:
            total = 1
        # Build batches of size total
        # Ensure the inner generator produces at least `total` examples
        try:
            # Prefer reconfiguring 'n' if attribute present
            if getattr(self._inner, "_n", 0) < total:
                self._inner._n = int(total)
        except Exception:
            pass
        batches = list(self._inner.batches(seed=seed, batch_size=total))
        if not batches:
            raise ValueError("seq2seq provider produced no examples")
        batch = batches[0]
        # Extract source tokens/masks and target ids for labels
        src_ids_list = [list(x) for x in batch.get("src_ids", [])][:total]
        src_mask_list = [list(x) for x in batch.get("src_mask", [])][:total]
        tgt_ids_list = [list(x) for x in batch.get("tgt_ids", [])][:total]
        # Right-pad/truncate to seq_len for runner compatibility
        pad_id = getattr(tokenizer, "pad_token_id", 0)

        def _pad(seq: list[int]) -> list[int]:
            if len(seq) < seq_len:
                return (seq + [pad_id] * (seq_len - len(seq)))[:seq_len]
            return seq[:seq_len]

        input_ids = [_pad(s) for s in src_ids_list]
        attention_masks = []
        for i, s in enumerate(input_ids):
            # Prefer src_mask if lengths align; otherwise infer from pad_id
            if i < len(src_mask_list) and len(src_mask_list[i]) == len(src_ids_list[i]):
                # Adjust length to seq_len
                m = src_mask_list[i]
                if len(m) < seq_len:
                    m = m + [0] * (seq_len - len(m))
                attention_masks.append([int(v) for v in m[:seq_len]])
            else:
                attention_masks.append([1 if t != pad_id else 0 for t in s])

        # Split into preview/final windows
        prev_ids = input_ids[:preview_n]
        prev_mask = attention_masks[:preview_n]
        fin_ids = input_ids[preview_n : preview_n + final_n]
        fin_mask = attention_masks[preview_n : preview_n + final_n]

        # Prepare label sequences (decoder targets) padded to seq_len
        def _pad_label(seq: list[int]) -> list[int]:
            if len(seq) < seq_len:
                return (seq + [-100] * (seq_len - len(seq)))[:seq_len]
            return seq[:seq_len]

        prev_labels = [_pad_label(s) for s in tgt_ids_list[:preview_n]]
        fin_labels = [
            _pad_label(s) for s in tgt_ids_list[preview_n : preview_n + final_n]
        ]
        # Save for runner to attach
        self.last_preview_labels = prev_labels
        self.last_final_labels = fin_labels

        preview_window = EvaluationWindow(prev_ids, prev_mask, list(range(preview_n)))
        final_window = EvaluationWindow(
            fin_ids, fin_mask, list(range(preview_n, preview_n + final_n))
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
        # Deterministic bounded synthetic examples; assume large enough for CI/release smokes
        n = int(target_total or 800)
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

    def info(self) -> dict[str, Any]:  # pragma: no cover - trivial
        return {"name": self.name, "type": "dataset_provider", "dataset": "seq2seq"}


# Registry for dataset providers
_PROVIDERS: dict[str, type] = {
    "wikitext2": WikiText2Provider,
    "synthetic": SyntheticProvider,
    "hf_text": HFTextProvider,
    "local_jsonl": LocalJSONLProvider,
    "seq2seq": Seq2SeqDataProvider,
    "hf_seq2seq": HFSeq2SeqProvider,
    "local_jsonl_pairs": LocalJSONLPairsProvider,
}


def get_provider(
    name: str, *, emit: EventEmitter | None = None, **kwargs: Any
) -> DatasetProvider:
    """
    Get a dataset provider by name.

    Args:
        name: Provider name ("wikitext2", "synthetic")
        emit: Optional event sink for dataset/provider logs.
        **kwargs: Provider-specific initialization parameters

    Returns:
        Initialized dataset provider

    Raises:
        ValidationError(E308): If provider name is not registered
    """
    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys())
        # Typed-only error for provider lookup
        raise _ValErr(
            code="E308",
            message="PROVIDER-NOT-FOUND: unknown dataset provider",
            details={"provider": name, "available": available},
        )

    provider_class = _PROVIDERS[name]
    init_kwargs = dict(kwargs)
    if emit is not None and name == "wikitext2":
        init_kwargs["emit"] = emit
    return provider_class(**init_kwargs)


def list_providers() -> list[str]:
    """List available dataset provider names."""
    return list(_PROVIDERS.keys())


def compute_window_hash(window: EvaluationWindow, include_data: bool = False) -> str:
    """
    Compute a deterministic hash of an evaluation window.

    Args:
        window: EvaluationWindow to hash
        include_data: Whether to include actual token data in hash

    Returns:
        Hex digest string of the window hash
    """
    hasher = hashlib.sha256()

    # Always include structural information
    hasher.update(str(len(window)).encode())
    hasher.update(str(sorted(window.indices)).encode())

    if include_data:
        # Include actual token sequences for data integrity checking
        for input_ids, attention_mask in zip(
            window.input_ids, window.attention_masks, strict=False
        ):
            hasher.update(str(input_ids).encode())
            hasher.update(str(attention_mask).encode())

    return hasher.hexdigest()


# Export public API
__all__ = [
    "DatasetProvider",
    "EvaluationWindow",
    "WikiText2Provider",
    "SyntheticProvider",
    "HFTextProvider",
    "LocalJSONLProvider",
    "get_provider",
    "list_providers",
    "compute_window_hash",
]
