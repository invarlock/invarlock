"""Shared helpers for hosted Hugging Face dataset providers."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from .data_support import _require_load_dataset, load_dataset_with_cache_fallback


def facade_attr(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("invarlock.eval.data_providers")
    if facade is None:
        return fallback
    return getattr(facade, name, fallback)


def require_dataset(message: str) -> None:
    require_fn = facade_attr("_require_load_dataset", _require_load_dataset)
    require_fn(message)


def load_dataset_from_facade(*args: Any, **kwargs: Any) -> Any:
    load_fn = facade_attr(
        "load_dataset_with_cache_fallback", load_dataset_with_cache_fallback
    )
    return load_fn(*args, **kwargs)


def field_value(row: Mapping[str, Any], field: str) -> Any:
    if not field:
        return None
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current
