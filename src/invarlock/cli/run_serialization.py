"""Config-like serialization helpers for run-shell orchestration."""

from __future__ import annotations

from typing import Any

from invarlock.core.run_policy import coerce_mapping as _coerce_mapping_impl


def _coerce_mapping(obj: object) -> dict[str, Any]:
    """Best-effort conversion of config-like objects to plain dicts."""
    return _coerce_mapping_impl(obj)


def _prune_none_values(value: Any) -> Any:
    """Recursively drop keys/items whose value is None."""

    if isinstance(value, dict):
        return {
            key: _prune_none_values(val)
            for key, val in value.items()
            if val is not None
        }
    if isinstance(value, list):
        return [_prune_none_values(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return tuple(_prune_none_values(item) for item in value if item is not None)
    return value


def _to_serialisable_dict(section: object) -> dict[str, Any]:
    """Coerce config fragments to plain dicts."""

    model_dump = getattr(section, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return _coerce_mapping(dumped)
    as_dict = getattr(section, "dict", None)
    if callable(as_dict):
        try:
            dumped = as_dict()
            if isinstance(dumped, dict):
                return _coerce_mapping(dumped)
        except (TypeError, ValueError):
            pass
    try:
        raw = getattr(section, "_data", None)
        if isinstance(raw, dict):
            return raw
    except AttributeError:
        pass
    if isinstance(section, dict):
        return section
    try:
        data = vars(section)
        if isinstance(data, dict) and isinstance(data.get("_data"), dict):
            return data["_data"]
        if isinstance(data, dict):
            return _coerce_mapping(_prune_none_values(data))
        return {}
    except TypeError:
        return {}
