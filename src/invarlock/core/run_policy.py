"""Deterministic run-policy and config-resolution helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from invarlock.core.auto_tuning import resolve_tier_policies

GUARD_OVERHEAD_THRESHOLD = 0.01

_MAPPING_COERCE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    TypeError,
    ValueError,
)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def coerce_mapping(obj: object) -> dict[str, Any]:
    """Convert config-like objects to plain dicts without hiding programming errors."""
    if isinstance(obj, dict):
        return obj
    try:
        raw = getattr(obj, "_data", None)
        if isinstance(raw, dict):
            return raw
    except _MAPPING_COERCE_EXCEPTIONS:
        raw = None
    try:
        dumped = obj.model_dump()  # type: ignore[attr-defined]
        if isinstance(dumped, dict):
            return dumped
    except _MAPPING_COERCE_EXCEPTIONS:
        dumped = None
    try:
        data = vars(obj)
        if isinstance(data, dict):
            return data
    except _MAPPING_COERCE_EXCEPTIONS:
        data = None
    return {}


def resolve_pm_acceptance_range(
    cfg: object | None,
    *,
    coerce_mapping_fn: Any | None = None,
) -> dict[str, float]:
    """Resolve primary-metric acceptance bounds from config with safe defaults."""
    if coerce_mapping_fn is None:
        coerce_mapping_fn = coerce_mapping
    base_min = 0.95
    base_max = 1.10

    cfg_map = coerce_mapping_fn(cfg) if cfg is not None else {}
    pm_section = cfg_map.get("primary_metric") if isinstance(cfg_map, dict) else {}
    pm_map = coerce_mapping_fn(pm_section)
    acceptance = pm_map.get("acceptance_range") if isinstance(pm_map, dict) else None
    cfg_min = None
    cfg_max = None
    if isinstance(acceptance, dict):
        cfg_min = _coerce_optional_float(acceptance.get("min"))
        cfg_max = _coerce_optional_float(acceptance.get("max"))

    has_explicit = any(v is not None for v in (cfg_min, cfg_max))
    if not has_explicit:
        return {}

    min_val = cfg_min if cfg_min is not None else base_min
    max_val = cfg_max if cfg_max is not None else base_max

    if min_val <= 0:
        min_val = base_min
    if max_val <= 0:
        max_val = base_max

    if max_val < min_val:
        max_val = min_val

    return {"min": float(min_val), "max": float(max_val)}


def resolve_pm_drift_band(
    cfg: object | None,
    *,
    coerce_mapping_fn: Any | None = None,
) -> dict[str, float]:
    """Resolve preview→final drift band from config with safe defaults."""
    if coerce_mapping_fn is None:
        coerce_mapping_fn = coerce_mapping
    base_min = 0.95
    base_max = 1.05

    cfg_map = coerce_mapping_fn(cfg) if cfg is not None else {}
    pm_section = cfg_map.get("primary_metric") if isinstance(cfg_map, dict) else {}
    pm_map = coerce_mapping_fn(pm_section)
    drift_band = pm_map.get("drift_band") if isinstance(pm_map, dict) else None
    cfg_min = None
    cfg_max = None
    if isinstance(drift_band, dict):
        cfg_min = _coerce_optional_float(drift_band.get("min"))
        cfg_max = _coerce_optional_float(drift_band.get("max"))
    elif isinstance(drift_band, list | tuple) and len(drift_band) == 2:
        cfg_min = _coerce_optional_float(drift_band[0])
        cfg_max = _coerce_optional_float(drift_band[1])

    has_explicit = any(v is not None for v in (cfg_min, cfg_max))
    if not has_explicit:
        return {}

    min_val = cfg_min if cfg_min is not None else base_min
    max_val = cfg_max if cfg_max is not None else base_max

    if min_val <= 0:
        min_val = base_min
    if max_val <= 0:
        max_val = base_max
    if min_val >= max_val:
        min_val, max_val = base_min, base_max

    return {"min": float(min_val), "max": float(max_val)}


def resolve_guard_overhead_threshold(
    cfg: object | None,
    *,
    default_threshold: float = GUARD_OVERHEAD_THRESHOLD,
    coerce_mapping_fn=coerce_mapping,
) -> float:
    """Resolve guard-overhead threshold from config with safe default fallback."""
    threshold = float(default_threshold)
    cfg_map = coerce_mapping_fn(cfg) if cfg is not None else {}
    pm_section = cfg_map.get("primary_metric") if isinstance(cfg_map, dict) else {}
    pm_map = coerce_mapping_fn(pm_section)
    candidate = pm_map.get("overhead_threshold") if isinstance(pm_map, dict) else None
    parsed = _coerce_optional_float(candidate)
    if parsed is not None and math.isfinite(parsed) and parsed >= 0.0:
        threshold = parsed
    return float(threshold)


def coerce_bool_like(value: Any) -> bool | None:
    """Best-effort bool coercion used for config policy toggles."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def resolve_skip_overhead_policy(
    cfg: object | None,
    *,
    coerce_mapping_fn=coerce_mapping,
) -> tuple[bool, str | None]:
    """Resolve overhead-skip policy from run/eval config context."""
    cfg_map = coerce_mapping_fn(cfg) if cfg is not None else {}
    if not isinstance(cfg_map, dict):
        return False, None
    ctx = coerce_mapping_fn(cfg_map.get("context"))
    run_ctx = coerce_mapping_fn(ctx.get("run")) if isinstance(ctx, dict) else {}
    eval_ctx = coerce_mapping_fn(ctx.get("eval")) if isinstance(ctx, dict) else {}

    run_val = coerce_bool_like(run_ctx.get("skip_overhead_check"))
    if run_val is not None:
        return bool(run_val), "config:context.run.skip_overhead_check"

    eval_val = coerce_bool_like(eval_ctx.get("skip_overhead_check"))
    if eval_val is not None:
        return bool(eval_val), "config:context.eval.skip_overhead_check"

    return False, None


def should_measure_overhead(
    profile_normalized: str,
    cfg: object | None,
    *,
    coerce_mapping_fn=coerce_mapping,
) -> tuple[bool, bool, str | None]:
    """Return overhead check policy resolved from profile + config context."""
    skip_overhead_cfg, skip_source = resolve_skip_overhead_policy(
        cfg, coerce_mapping_fn=coerce_mapping_fn
    )
    enforce_profile = profile_normalized in {"ci", "release"}
    skip_overhead = bool(skip_overhead_cfg and enforce_profile)
    measure_guard_overhead = bool(enforce_profile and not skip_overhead)
    source = skip_source if skip_overhead else None
    return measure_guard_overhead, skip_overhead, source


def resolve_pm_min_tokens_target(
    *,
    tier: str | None,
    profile: str | None,
) -> int:
    """Resolve the minimum PM token target from tier policy."""
    resolved = resolve_tier_policies((tier or "balanced").lower(), profile=profile)
    metrics = resolved.get("metrics", {}) if isinstance(resolved, dict) else {}
    pm_ratio = metrics.get("pm_ratio", {}) if isinstance(metrics, dict) else {}
    try:
        return int(pm_ratio.get("min_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def choose_dataset_split(
    *,
    requested: str | None,
    available: list[str] | None,
    split_aliases: Sequence[str] = ("validation", "val", "dev", "eval", "test"),
) -> tuple[str, bool]:
    """Choose a dataset split deterministically."""
    if isinstance(requested, str) and requested:
        return requested, False
    avail = list(available) if isinstance(available, list) and available else []
    if avail:
        for cand in split_aliases:
            if cand in avail:
                return cand, True
        return sorted(avail)[0], True
    return "validation", True
