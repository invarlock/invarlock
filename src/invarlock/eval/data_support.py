"""
Evaluation data runtime support.

Owns dependency detection, lazy dataset loading, and local file signatures.
"""

import hashlib
import importlib.util
import json
import logging
import math
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from glob import glob as _glob
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, TypeAlias

from invarlock.core.exceptions import DependencyError as _DepErr

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
LOGGER = logging.getLogger(__name__)

_WORKER_INIT_IMPORT_ERRORS = (
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
)


def deterministic_worker_init_fn(worker_id: int, *, base_seed: int = 0) -> None:
    """Best-effort deterministic worker initializer."""
    try:
        import random

        random.seed((base_seed ^ (worker_id + 17)) & 0x7FFFFFFF)
    except _WORKER_INIT_IMPORT_ERRORS:
        pass
    try:
        import numpy as _np

        _np.random.seed(((base_seed + 97) ^ (worker_id * 131)) & 0x7FFFFFFF)
    except _WORKER_INIT_IMPORT_ERRORS:
        pass
    try:  # pragma: no cover - torch may be unavailable in CI
        import torch as _torch

        _torch.manual_seed((base_seed * 1009 + worker_id * 7919) & 0x7FFFFFFF)
        if hasattr(_torch.cuda, "manual_seed_all"):
            _torch.cuda.manual_seed_all(
                (base_seed * 1013 + worker_id * 7951) & 0x7FFFFFFF
            )
    except _WORKER_INIT_IMPORT_ERRORS:
        pass


def deterministic_shards(n: int, *, num_workers: int) -> list[list[int]]:
    """Return a deterministic partition of `range(n)` across workers."""
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
        """Return a reproducibility digest."""

    def batches(self, *, seed: int, batch_size: int) -> Iterable[dict[str, Any]]:
        """Yield task-appropriate batches."""


class EvaluationWindow(NamedTuple):
    """A window of tokenized samples for evaluation."""

    input_ids: list[list[int]]
    attention_masks: list[list[int]]
    indices: list[int]

    def __len__(self) -> int:
        return len(self.input_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_masks": self.attention_masks,
            "indices": self.indices,
            "length": len(self.input_ids),
        }


def split_window_by_index(
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


def split_labels_by_index(
    labels: list[list[int]],
    indices: list[int],
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


def compute_window_hash(window: EvaluationWindow, include_data: bool = False) -> str:
    """Compute a deterministic hash of an evaluation window."""
    hasher = hashlib.sha256()

    hasher.update(str(len(window)).encode())
    hasher.update(str(sorted(window.indices)).encode())

    if include_data:
        for input_ids, attention_mask in zip(
            window.input_ids, window.attention_masks, strict=False
        ):
            hasher.update(str(input_ids).encode())
            hasher.update(str(attention_mask).encode())

    return hasher.hexdigest()


DatasetDiagnosticSeverity: TypeAlias = Literal["info", "warning", "error"]  # noqa: UP040
DatasetDiagnosticCategory: TypeAlias = Literal["dataset", "provider", "window"]  # noqa: UP040


@dataclass(frozen=True)
class DatasetDiagnostic:
    kind: str
    message: str
    severity: DatasetDiagnosticSeverity = "info"
    metadata: dict[str, Any] = field(default_factory=dict)
    code: str | None = None
    category: DatasetDiagnosticCategory | None = None

    def __post_init__(self) -> None:
        if self.code is None:
            object.__setattr__(self, "code", str(self.kind))
        if self.category is None:
            kind = str(self.kind)
            category = kind.split(".", 1)[0] if "." in kind else "dataset"
            if category not in {"dataset", "provider", "window"}:
                category = "dataset"
            object.__setattr__(self, "category", category)


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


def _is_hf_datasets_cache_lock_error(exc: BaseException) -> bool:
    message = " ".join(str(part) for part in exc.args if part).lower()
    if not message:
        message = str(exc).lower()
    return (
        ".lock" in message
        and ("operation not permitted" in message or "permission denied" in message)
        and ("huggingface" in message or "datasets" in message)
    )


def _default_invarlock_datasets_cache_dir() -> Path:
    configured = os.getenv("INVARLOCK_HF_DATASETS_CACHE", "").strip()
    if configured:
        return Path(configured).expanduser()
    cache_home = os.getenv("XDG_CACHE_HOME", "").strip()
    if cache_home:
        return Path(cache_home).expanduser() / "invarlock" / "hf_datasets"
    return Path.home() / ".cache" / "invarlock" / "hf_datasets"


def _ensure_invarlock_datasets_cache_dir() -> Path:
    preferred = _default_invarlock_datasets_cache_dir()
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.mkdtemp(prefix="invarlock_hf_datasets_"))
        LOGGER.warning(
            "Falling back to temporary datasets cache at %s after failing to create %s",
            fallback,
            preferred,
        )
        return fallback


def load_dataset_with_cache_fallback(
    *args: Any,
    cache_dir: str | None = None,
    **kwargs: Any,
) -> Any:
    load_dataset_fn = _require_load_dataset(
        "DEPENDENCY-MISSING: datasets library required for Hugging Face dataset loading"
    )
    chosen_cache_dir = cache_dir
    try:
        return load_dataset_fn(*args, cache_dir=chosen_cache_dir, **kwargs)
    except (OSError, PermissionError) as exc:
        env_cache_dir = os.getenv("HF_DATASETS_CACHE", "").strip()
        if (
            chosen_cache_dir
            or env_cache_dir
            or not _is_hf_datasets_cache_lock_error(exc)
        ):
            raise
        fallback_dir = _ensure_invarlock_datasets_cache_dir()
        LOGGER.warning(
            "Retrying datasets load with writable InvarLock cache %s after shared cache lock error: %s",
            fallback_dir,
            exc,
        )
        return load_dataset_fn(*args, cache_dir=str(fallback_dir), **kwargs)


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


def estimate_wikitext2_capacity(
    *,
    load_fn: Callable[..., list[str]],
    collect_tokenized_samples_fn: Callable[
        [list[str], list[int], Any, int], list[tuple[int, list[int], list[int], int]]
    ],
    tokenizer: Any,
    seq_len: int,
    stride: int,
    split: str = "validation",
    target_total: int | None = None,
    fast_mode: bool = False,
) -> dict[str, Any]:
    texts = load_fn(split=split, max_samples=2000)
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
        requested_total = int(target_total or 0)
        approx_available = (
            max(base_available, requested_total)
            if requested_total > 0
            else base_available
        )
        approx_available = int(max(approx_available, 0))
        return {
            "total_tokens": int(approx_available * seq_len),
            "available_nonoverlap": approx_available,
            "available_unique": approx_available,
            "dedupe_rate": 0.0,
            "stride": int(stride),
            "seq_len": int(seq_len),
            "candidate_unique": approx_available,
            "candidate_limit": approx_available,
        }

    tokenized = collect_tokenized_samples_fn(
        texts, list(range(len(texts))), tokenizer, seq_len
    )
    total_tokens = sum(item[3] for item in tokenized)
    available_nonoverlap = len(tokenized)
    unique_sequences = {
        tuple(
            int(tok_id)
            for tok_id, mask in zip(input_ids, attention_mask, strict=False)
            if mask
        )
        for _, input_ids, attention_mask, _ in tokenized
    }
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
        tokenized_subset = collect_tokenized_samples_fn(
            texts, list(range(candidate_limit)), tokenizer, seq_len
        )
        subset_signatures = {
            tuple(
                int(tok) for tok, mask in zip(entry[1], entry[2], strict=False) if mask
            )
            for entry in tokenized_subset
        }
        candidate_unique = len(subset_signatures)

    result: dict[str, Any] = {
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


def score_candidates_byte_ngram(
    candidates: list[dict[str, Any]],
    *,
    order: int,
    pad_token: int,
    alpha: float,
) -> dict[str, Any] | None:
    if not candidates:
        return None

    order = max(1, int(order))
    pad_token = int(pad_token)
    alpha = float(alpha)
    vocab_size = pad_token + 1
    context_width = max(order - 1, 0)
    context_modulus = vocab_size ** max(context_width - 1, 0)

    def initial_context_key() -> int:
        key = 0
        for _ in range(context_width):
            key = (key * vocab_size) + pad_token
        return key

    def next_context_key(current_key: int, token: int) -> int:
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
        context_key = initial_context_key()
        for token in byte_values:
            context_counts[context_key] += 1
            ngram_counts[(context_key * vocab_size) + int(token)] += 1
            context_key = next_context_key(context_key, int(token))

    total_tokens = 0
    for candidate, byte_values in zip(candidates, sequences, strict=False):
        loss_sum = 0.0
        token_count = 0
        context_key = initial_context_key()
        for token in byte_values:
            context_count = context_counts.get(context_key, 0)
            ngram_key = (context_key * vocab_size) + int(token)
            ngram_count = ngram_counts.get(ngram_key, 0)
            prob = (ngram_count + alpha) / (context_count + alpha * vocab_size)
            loss_sum += -math.log(prob)
            token_count += 1
            context_key = next_context_key(context_key, int(token))
        candidate["difficulty"] = loss_sum / max(token_count, 1)
        total_tokens += token_count

    elapsed = max(time.perf_counter() - start_time, 1e-9)
    tokens_per_sec = total_tokens / elapsed if total_tokens else 0.0
    return {
        "mode": "byte_ngram",
        "order": order,
        "vocab_size": vocab_size,
        "tokens_processed": total_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_per_sec,
    }


def resolve_local_jsonl_files(
    *,
    file: str | None = None,
    path: str | None = None,
    data_files: str | list[str] | None = None,
) -> list[Path]:
    files: list[Path] = []
    if isinstance(file, str) and file:
        p = Path(file)
        if p.exists() and p.is_file():
            files.append(p)
    if isinstance(path, str) and path:
        p = Path(path)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
    if isinstance(data_files, str) and data_files:
        files.extend(Path(p) for p in _glob(data_files))
    elif isinstance(data_files, list):
        for item in data_files:
            try:
                pp = Path(str(item))
                if pp.exists() and pp.is_file():
                    files.append(pp)
            except (AttributeError, OSError, TypeError, ValueError):
                continue
    seen: set[str] = set()
    uniq: list[Path] = []
    for file_path in files:
        resolved = file_path.resolve().as_posix()
        if resolved not in seen:
            seen.add(resolved)
            uniq.append(file_path)
    return uniq


def iter_local_jsonl_objects(files: list[Path]) -> Iterator[dict[str, Any]]:
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except (OSError, UnicodeDecodeError):
            continue


def load_local_jsonl_texts(
    files: list[Path], *, text_field: str, max_samples: int
) -> list[str]:
    texts: list[str] = []
    for obj in iter_local_jsonl_objects(files):
        value = obj.get(text_field)
        if isinstance(value, str) and value.strip():
            texts.append(value)
            if len(texts) >= max_samples:
                return texts
    return texts


def load_local_jsonl_pairs(
    files: list[Path], *, src_field: str, tgt_field: str, max_samples: int
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for obj in iter_local_jsonl_objects(files):
        src = obj.get(src_field)
        tgt = obj.get(tgt_field)
        if (
            isinstance(src, str)
            and src.strip()
            and isinstance(tgt, str)
            and tgt.strip()
        ):
            pairs.append((src, tgt))
            if len(pairs) >= max_samples:
                return pairs
    return pairs


def local_jsonl_cache_key(
    files: list[Path], *, field_names: tuple[str, ...], max_samples: int
) -> tuple[Any, ...]:
    return (
        _local_files_signature(files),
        tuple(field_names),
        (int(max_samples),),
    )


__all__ = [
    "DatasetDiagnostic",
    "DatasetDiagnosticCategory",
    "DatasetDiagnosticSeverity",
    "EvaluationProvider",
    "EvaluationWindow",
    "HAS_DATASETS",
    "HAS_TORCH",
    "compute_window_hash",
    "deterministic_shards",
    "deterministic_worker_init_fn",
    "estimate_wikitext2_capacity",
    "iter_local_jsonl_objects",
    "_get_load_dataset",
    "_require_load_dataset",
    "_is_hf_datasets_cache_lock_error",
    "_local_files_signature",
    "load_local_jsonl_pairs",
    "load_local_jsonl_texts",
    "load_dataset_with_cache_fallback",
    "load_dataset",
    "local_jsonl_cache_key",
    "resolve_local_jsonl_files",
    "score_candidates_byte_ngram",
    "split_labels_by_index",
    "split_window_by_index",
]
