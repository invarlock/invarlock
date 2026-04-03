"""
Evaluation data runtime support.

Owns dependency detection, lazy dataset loading, and local file signatures.
"""

import importlib.util
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

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


__all__ = [
    "DatasetDiagnostic",
    "DatasetDiagnosticCategory",
    "DatasetDiagnosticSeverity",
    "HAS_DATASETS",
    "HAS_TORCH",
    "_get_load_dataset",
    "_require_load_dataset",
    "_is_hf_datasets_cache_lock_error",
    "_local_files_signature",
    "load_dataset_with_cache_fallback",
    "load_dataset",
]
