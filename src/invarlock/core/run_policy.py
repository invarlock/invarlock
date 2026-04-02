"""Deterministic run-policy and config-resolution helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from invarlock.core.auto_tuning import resolve_tier_policies
from invarlock.core.exceptions import ConfigError

GUARD_OVERHEAD_THRESHOLD = 0.01


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


def _raise_config_error(path: str, value: Any, message: str) -> None:
    raise ConfigError(
        code="E002",
        message=message,
        details={"path": path, "value": value},
    )


def coerce_mapping(obj: object) -> dict[str, Any]:
    """Convert config-like objects to plain dicts without hiding programming errors."""
    if isinstance(obj, dict):
        return obj
    try:
        raw = getattr(obj, "_data", None)
    except AttributeError:
        raw = None
    if isinstance(raw, dict):
        return raw
    dumped = getattr(obj, "model_dump", None)
    if callable(dumped):
        result = dumped()
        if isinstance(result, dict):
            return result
    try:
        data = obj.__dict__
    except (AttributeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


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
    if acceptance is None:
        return {}
    if not isinstance(acceptance, dict):
        _raise_config_error(
            "primary_metric.acceptance_range",
            acceptance,
            "primary_metric.acceptance_range must be a mapping with optional min/max bounds.",
        )
    cfg_min = None
    cfg_max = None
    if "min" in acceptance:
        cfg_min = _coerce_optional_float(acceptance.get("min"))
        if cfg_min is None:
            _raise_config_error(
                "primary_metric.acceptance_range.min",
                acceptance.get("min"),
                "primary_metric.acceptance_range.min must be a positive finite number.",
            )
    if "max" in acceptance:
        cfg_max = _coerce_optional_float(acceptance.get("max"))
        if cfg_max is None:
            _raise_config_error(
                "primary_metric.acceptance_range.max",
                acceptance.get("max"),
                "primary_metric.acceptance_range.max must be a positive finite number.",
            )

    has_explicit = any(v is not None for v in (cfg_min, cfg_max))
    if not has_explicit:
        return {}

    min_val = cfg_min if cfg_min is not None else base_min
    max_val = cfg_max if cfg_max is not None else base_max

    if min_val <= 0:
        _raise_config_error(
            "primary_metric.acceptance_range.min",
            min_val,
            "primary_metric.acceptance_range.min must be greater than zero.",
        )
    if max_val <= 0:
        _raise_config_error(
            "primary_metric.acceptance_range.max",
            max_val,
            "primary_metric.acceptance_range.max must be greater than zero.",
        )

    if max_val < min_val:
        _raise_config_error(
            "primary_metric.acceptance_range",
            acceptance,
            "primary_metric.acceptance_range.max must be greater than or equal to min.",
        )

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
    if drift_band is None:
        return {}
    cfg_min = None
    cfg_max = None
    if isinstance(drift_band, dict):
        if "min" in drift_band:
            cfg_min = _coerce_optional_float(drift_band.get("min"))
            if cfg_min is None:
                _raise_config_error(
                    "primary_metric.drift_band.min",
                    drift_band.get("min"),
                    "primary_metric.drift_band.min must be a positive finite number.",
                )
        if "max" in drift_band:
            cfg_max = _coerce_optional_float(drift_band.get("max"))
            if cfg_max is None:
                _raise_config_error(
                    "primary_metric.drift_band.max",
                    drift_band.get("max"),
                    "primary_metric.drift_band.max must be a positive finite number.",
                )
    elif isinstance(drift_band, list | tuple) and len(drift_band) == 2:
        cfg_min = _coerce_optional_float(drift_band[0])
        cfg_max = _coerce_optional_float(drift_band[1])
        if cfg_min is None or cfg_max is None:
            _raise_config_error(
                "primary_metric.drift_band",
                drift_band,
                "primary_metric.drift_band list form must contain two positive finite numbers.",
            )
    else:
        _raise_config_error(
            "primary_metric.drift_band",
            drift_band,
            "primary_metric.drift_band must be a mapping or a two-item list/tuple.",
        )

    has_explicit = any(v is not None for v in (cfg_min, cfg_max))
    if not has_explicit:
        return {}

    min_val = cfg_min if cfg_min is not None else base_min
    max_val = cfg_max if cfg_max is not None else base_max

    if min_val <= 0:
        _raise_config_error(
            "primary_metric.drift_band.min",
            min_val,
            "primary_metric.drift_band.min must be greater than zero.",
        )
    if max_val <= 0:
        _raise_config_error(
            "primary_metric.drift_band.max",
            max_val,
            "primary_metric.drift_band.max must be greater than zero.",
        )
    if min_val >= max_val:
        _raise_config_error(
            "primary_metric.drift_band",
            drift_band,
            "primary_metric.drift_band.min must be less than max.",
        )

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
    if candidate is None:
        return float(threshold)
    parsed = _coerce_optional_float(candidate)
    if parsed is None or not math.isfinite(parsed) or parsed < 0.0:
        _raise_config_error(
            "primary_metric.overhead_threshold",
            candidate,
            "primary_metric.overhead_threshold must be a non-negative finite number.",
        )
    assert parsed is not None
    threshold = float(parsed)
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
        min_tokens = int(pm_ratio.get("min_tokens", 0) or 0)
    except (TypeError, ValueError):
        _raise_config_error(
            "tier_policies.metrics.pm_ratio.min_tokens",
            pm_ratio.get("min_tokens"),
            "Resolved tier policy min_tokens must be an integer.",
        )
    if min_tokens < 0:
        _raise_config_error(
            "tier_policies.metrics.pm_ratio.min_tokens",
            min_tokens,
            "Resolved tier policy min_tokens must be non-negative.",
        )
    return min_tokens


def choose_dataset_split(
    *,
    requested: str | None,
    available: list[str] | None,
    split_aliases: Sequence[str] = ("validation", "val", "dev", "eval", "test"),
) -> tuple[str, bool]:
    """Choose a dataset split deterministically."""
    if isinstance(requested, str):
        requested_text = str(requested)
        if requested_text:
            return requested_text, False
    avail = list(available) if isinstance(available, list) and available else []
    if avail:
        for cand in split_aliases:
            if cand in avail:
                return cand, True
        return sorted(avail)[0], True
    return "validation", True
