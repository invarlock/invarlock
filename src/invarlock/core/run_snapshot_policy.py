from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def resolve_snapshot_config(
    context: object | None,
    *,
    to_serialisable_dict_fn: Callable[[object], Any],
) -> dict[str, Any]:
    """Extract a plain snapshot policy mapping from run context."""
    try:
        context_map = to_serialisable_dict_fn(context or {})
    except (AttributeError, TypeError, ValueError):
        return {}
    if not isinstance(context_map, dict):
        return {}
    try:
        snapshot_map = to_serialisable_dict_fn(context_map.get("snapshot", {}))
    except (AttributeError, TypeError, ValueError):
        return {}
    return snapshot_map if isinstance(snapshot_map, dict) else {}


def estimate_model_bytes(model: Any) -> int:
    """Best-effort estimate of parameter + buffer footprint."""
    total = 0
    try:
        for _, param in getattr(model, "named_parameters", lambda: [])():
            try:
                total += int(param.element_size() * param.nelement())
            except (AttributeError, TypeError, ValueError):
                pass
        for _, buffer in getattr(model, "named_buffers", lambda: [])():
            try:
                total += int(buffer.element_size() * buffer.nelement())
            except (AttributeError, TypeError, ValueError):
                pass
    except (AttributeError, TypeError):
        return 0
    return total


def _parse_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed


def _requested_snapshot_mode(
    requested_mode: str,
    *,
    supports_bytes: bool,
    supports_chunked: bool,
) -> str:
    if requested_mode == "bytes" and supports_bytes:
        return "bytes"
    if requested_mode == "chunked" and supports_chunked:
        return "chunked"
    if supports_bytes:
        return "bytes"
    if supports_chunked:
        return "chunked"
    return "reload"


def choose_snapshot_mode(
    *,
    snapshot_config: Mapping[str, Any] | None,
    env_mode: str | None,
    supports_bytes: bool,
    supports_chunked: bool,
    estimated_model_mb: float,
    available_ram_mb: float,
    disk_free_mb: float,
    env_ram_fraction: str | None = None,
    env_threshold_mb: str | None = None,
) -> str:
    """Choose bytes/chunked/reload snapshot mode deterministically."""
    cfg_snapshot = dict(snapshot_config or {})
    cfg_mode = str(cfg_snapshot.get("mode", "")).lower()
    mode_env = str(env_mode or "auto").lower()

    if cfg_mode in {"bytes", "chunked"}:
        return _requested_snapshot_mode(
            cfg_mode,
            supports_bytes=supports_bytes,
            supports_chunked=supports_chunked,
        )
    if mode_env in {"bytes", "chunked"}:
        return _requested_snapshot_mode(
            mode_env,
            supports_bytes=supports_bytes,
            supports_chunked=supports_chunked,
        )

    frac = 0.4
    if cfg_snapshot.get("ram_fraction") is not None:
        frac = _parse_float(cfg_snapshot.get("ram_fraction"), frac)
    elif env_ram_fraction is not None:
        frac = _parse_float(env_ram_fraction, frac)
    frac = max(0.0, min(frac, 1.0))

    if available_ram_mb > 0:
        threshold_mb = float(available_ram_mb) * frac
    elif cfg_snapshot.get("threshold_mb") is not None:
        threshold_mb = _parse_float(cfg_snapshot.get("threshold_mb"), 768.0)
    else:
        threshold_mb = _parse_float(env_threshold_mb, 768.0)

    margin = 1.2
    if cfg_snapshot.get("disk_free_margin_ratio") is not None:
        margin = _parse_float(cfg_snapshot.get("disk_free_margin_ratio"), margin)

    disk_has_room = disk_free_mb <= 0.0 or estimated_model_mb * margin <= disk_free_mb

    if supports_chunked and estimated_model_mb >= threshold_mb and disk_has_room:
        return "chunked"

    if supports_bytes:
        if (
            supports_chunked
            and available_ram_mb > 0
            and estimated_model_mb >= max(64.0, float(available_ram_mb) * 0.8)
            and disk_has_room
        ):
            return "chunked"
        return "bytes"

    if supports_chunked:
        return "chunked"

    return "reload"
