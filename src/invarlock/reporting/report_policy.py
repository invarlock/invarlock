"""Policy-resolution helpers for evaluation report assembly."""

from __future__ import annotations

import math
from typing import Any

TIER_RATIO_LIMITS: dict[str, float] = {
    "conservative": 1.05,
    "balanced": 1.10,
    "aggressive": 1.20,
    "none": 1.10,
}

PM_DRIFT_BAND_DEFAULT: tuple[float, float] = (0.95, 1.05)


def resolve_pm_acceptance_range_from_report(
    report: dict[str, Any] | None,
) -> dict[str, float]:
    """Resolve primary-metric acceptance bounds from report context."""

    base_min = 0.95
    base_max = 1.10

    def _safe_float(val: Any) -> float | None:
        try:
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    cfg_min = None
    cfg_max = None
    ctx = report.get("context") if isinstance(report, dict) else None
    if isinstance(ctx, dict):
        pm_ctx = (
            ctx.get("primary_metric")
            if isinstance(ctx.get("primary_metric"), dict)
            else {}
        )
        if isinstance(pm_ctx, dict):
            acceptance_range = pm_ctx.get("acceptance_range")
            if isinstance(acceptance_range, dict):
                cfg_min = _safe_float(acceptance_range.get("min"))
                cfg_max = _safe_float(acceptance_range.get("max"))
    has_explicit = any(v is not None for v in (cfg_min, cfg_max))
    if not has_explicit:
        return {}

    min_val = cfg_min if cfg_min is not None else base_min
    max_val = cfg_max if cfg_max is not None else base_max

    try:
        if min_val is not None and min_val <= 0:
            min_val = base_min
    except Exception:
        min_val = base_min
    try:
        if max_val is not None and max_val <= 0:
            max_val = base_max
    except Exception:
        max_val = base_max

    try:
        if max_val is not None and min_val is not None and max_val < min_val:
            max_val = min_val
    except Exception:
        max_val = base_max

    return {"min": float(min_val), "max": float(max_val)}


def resolve_pm_drift_band_from_report(
    report: dict[str, Any] | None,
    *,
    drift_band_default: tuple[float, float] = (0.95, 1.05),
) -> dict[str, float]:
    """Resolve preview→final drift band from report context."""

    base_min, base_max = drift_band_default

    def _safe_float(val: Any) -> float | None:
        try:
            if val is None:
                return None
            out = float(val)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    cfg_min = None
    cfg_max = None

    ctx = report.get("context") if isinstance(report, dict) else None
    if isinstance(ctx, dict):
        pm_ctx = ctx.get("primary_metric")
        if isinstance(pm_ctx, dict):
            band = pm_ctx.get("drift_band")
            if isinstance(band, dict):
                cfg_min = _safe_float(band.get("min"))
                cfg_max = _safe_float(band.get("max"))
            elif isinstance(band, list | tuple) and len(band) == 2:
                cfg_min = _safe_float(band[0])
                cfg_max = _safe_float(band[1])
    has_explicit = any(v is not None for v in (cfg_min, cfg_max))
    if not has_explicit:
        return {}

    min_val = cfg_min if cfg_min is not None else base_min
    max_val = cfg_max if cfg_max is not None else base_max

    try:
        if min_val is not None and min_val <= 0:
            min_val = base_min
    except Exception:
        min_val = base_min
    try:
        if max_val is not None and max_val <= 0:
            max_val = base_max
    except Exception:
        max_val = base_max
    try:
        if min_val is not None and max_val is not None and min_val >= max_val:
            min_val, max_val = base_min, base_max
    except Exception:
        min_val, max_val = base_min, base_max

    return {"min": float(min_val), "max": float(max_val)}


def resolve_tiny_relax_from_report(report: dict[str, Any] | None) -> bool:
    """Resolve tiny-relax mode from report context policy fields."""

    def _coerce_bool_like(value: Any) -> bool | None:
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

    if not isinstance(report, dict):
        return False

    ctx = report.get("context")
    if isinstance(ctx, dict):
        run_ctx = ctx.get("run")
        if isinstance(run_ctx, dict):
            run_val = _coerce_bool_like(run_ctx.get("tiny_relax"))
            if run_val is not None:
                return bool(run_val)
        eval_ctx = ctx.get("eval")
        if isinstance(eval_ctx, dict):
            eval_val = _coerce_bool_like(eval_ctx.get("tiny_relax"))
            if eval_val is not None:
                return bool(eval_val)

    return False
