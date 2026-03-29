"""
Evaluation data runtime support.

Owns dependency detection, lazy dataset loading, and local file signatures.
"""

import importlib.util
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class DatasetDiagnostic:
    kind: str
    message: str
    severity: str = "info"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tag(self) -> str:
        tag = self.metadata.get("tag", self.kind)
        return str(tag).upper()

    @property
    def emoji(self) -> str | None:
        emoji = self.metadata.get("emoji")
        if isinstance(emoji, str) and emoji:
            return emoji
        return None


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
    "HAS_DATASETS",
    "HAS_TORCH",
    "_get_load_dataset",
    "_require_load_dataset",
    "_local_files_signature",
    "load_dataset",
]
