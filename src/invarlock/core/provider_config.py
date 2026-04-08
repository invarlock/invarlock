from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ITEMS_CALL_ERRORS = (AttributeError, TypeError, ValueError)


def _mapping_items(value: Any) -> list[tuple[Any, Any]]:
    if isinstance(value, Mapping):
        return list(value.items())
    data = getattr(value, "_data", None)
    if isinstance(data, Mapping):
        return list(data.items())
    items = getattr(value, "items", None)
    if callable(items):
        try:
            return list(items())
        except _ITEMS_CALL_ERRORS:
            return []
    return []


def resolve_provider_kind_and_kwargs(
    provider_value: Any,
) -> tuple[str | None, dict[str, Any]]:
    """Return canonical provider kind plus explicit provider kwargs."""

    if isinstance(provider_value, str):
        normalized = provider_value.strip()
        return (normalized or None), {}

    provider_items = _mapping_items(provider_value)
    if not provider_items:
        return None, {}

    provider_kwargs: dict[str, Any] = {}
    provider_kind: str | None = None
    for raw_key, raw_value in provider_items:
        key = str(raw_key)
        if key == "kind":
            candidate = str(raw_value).strip() if raw_value is not None else ""
            provider_kind = candidate or None
            continue
        if raw_value is None or raw_value == "":
            continue
        provider_kwargs[key] = raw_value
    return provider_kind, provider_kwargs


__all__ = ["resolve_provider_kind_and_kwargs"]
