from __future__ import annotations

from functools import lru_cache
from typing import Any

from invarlock.public_contracts import load_json_contract


class MetricKindContractError(RuntimeError):
    """Raised when the shipped metric-kind contract cannot be loaded."""


@lru_cache(maxsize=1)
def load_metric_kind_catalog() -> frozenset[str]:
    try:
        payload = load_json_contract("metric_kinds.json")
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        NotADirectoryError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise MetricKindContractError(
            f"Failed to load metric kind contract: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise MetricKindContractError(
            "metric_kinds.json must decode to a non-empty JSON array of strings."
        )
    kinds = {
        str(item).strip().lower()
        for item in payload
        if isinstance(item, str) and item.strip()
    }
    if not kinds:
        raise MetricKindContractError(
            "metric_kinds.json must contain at least one concrete metric kind."
        )
    return frozenset(kinds)


def normalize_metric_kind(value: Any, *, allow_auto: bool = False) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized == "auto":
        return None if allow_auto else None
    supported_kinds = load_metric_kind_catalog()
    if normalized not in supported_kinds:
        supported = ", ".join(sorted(supported_kinds))
        raise ValueError(
            f"Unsupported metric kind '{value}'. Supported kinds: {supported}"
        )
    return normalized


def is_known_metric_kind(value: Any) -> bool:
    try:
        return normalize_metric_kind(value) is not None
    except (MetricKindContractError, ValueError):
        return False


def is_ppl_metric_kind(value: Any) -> bool:
    try:
        normalized = normalize_metric_kind(value)
    except (MetricKindContractError, ValueError):
        return False
    return bool(normalized and normalized.startswith("ppl"))


__all__ = [
    "MetricKindContractError",
    "is_known_metric_kind",
    "is_ppl_metric_kind",
    "load_metric_kind_catalog",
    "normalize_metric_kind",
]
