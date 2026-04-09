"""Policy-resolution helpers for evaluation report assembly."""

from __future__ import annotations

import math
from typing import Any

from .report_policy_parsing import coerce_bool_like

TIER_RATIO_LIMITS: dict[str, float] = {
    "conservative": 1.05,
    "balanced": 1.10,
    "aggressive": 1.20,
    "none": 1.10,
}

PM_DRIFT_BAND_DEFAULT: tuple[float, float] = (0.95, 1.05)
_PARSE_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _coerce_finite_float_local(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except _PARSE_EXCEPTIONS:
        return None
    try:
        if not math.isfinite(parsed):
            return None
    except _PARSE_EXCEPTIONS:
        return None
    return parsed


def _primary_metric_policy_source(
    report: dict[str, Any] | None,
    *,
    meta_key: str,
) -> Any:
    ctx = report.get("context") if isinstance(report, dict) else None
    if isinstance(ctx, dict):
        pm_ctx = ctx.get("primary_metric")
        if isinstance(pm_ctx, dict):
            source = pm_ctx.get(meta_key.removeprefix("pm_"))
            if source is not None:
                return source

    meta = report.get("meta") if isinstance(report, dict) else None
    if isinstance(meta, dict):
        return meta.get(meta_key)
    return None


def resolve_pm_acceptance_range_from_report(
    report: dict[str, Any] | None,
) -> dict[str, float]:
    """Resolve primary-metric acceptance bounds from report context."""

    base_min = 0.95
    base_max = 1.10

    cfg_min = None
    cfg_max = None
    acceptance_range = _primary_metric_policy_source(
        report,
        meta_key="pm_acceptance_range",
    )
    if isinstance(acceptance_range, dict):
        cfg_min = _coerce_finite_float_local(acceptance_range.get("min"))
        cfg_max = _coerce_finite_float_local(acceptance_range.get("max"))
    has_explicit = any(v is not None for v in (cfg_min, cfg_max))
    if not has_explicit:
        return {}

    min_val = cfg_min if cfg_min is not None else base_min
    max_val = cfg_max if cfg_max is not None else base_max

    try:
        min_nonpositive = min_val <= 0
    except _PARSE_EXCEPTIONS:
        min_nonpositive = True
    if min_nonpositive:
        min_val = base_min
    try:
        max_nonpositive = max_val <= 0
    except _PARSE_EXCEPTIONS:
        max_nonpositive = True
    if max_nonpositive:
        max_val = base_max
    try:
        inverted = max_val < min_val
    except _PARSE_EXCEPTIONS:
        max_val = base_max
        inverted = False
    if inverted:
        max_val = min_val

    return {"min": float(min_val), "max": float(max_val)}


def resolve_pm_drift_band_from_report(
    report: dict[str, Any] | None,
    *,
    drift_band_default: tuple[float, float] = (0.95, 1.05),
) -> dict[str, float]:
    """Resolve preview→final drift band from report context."""

    base_min, base_max = drift_band_default

    cfg_min = None
    cfg_max = None

    band = _primary_metric_policy_source(report, meta_key="pm_drift_band")
    if isinstance(band, dict):
        cfg_min = _coerce_finite_float_local(band.get("min"))
        cfg_max = _coerce_finite_float_local(band.get("max"))
    elif isinstance(band, list | tuple) and len(band) == 2:
        cfg_min = _coerce_finite_float_local(band[0])
        cfg_max = _coerce_finite_float_local(band[1])
    has_explicit = any(v is not None for v in (cfg_min, cfg_max))
    if not has_explicit:
        return {}

    min_val = cfg_min if cfg_min is not None else base_min
    max_val = cfg_max if cfg_max is not None else base_max

    try:
        min_nonpositive = min_val <= 0
    except _PARSE_EXCEPTIONS:
        min_nonpositive = True
    if min_nonpositive:
        min_val = base_min
    try:
        max_nonpositive = max_val <= 0
    except _PARSE_EXCEPTIONS:
        max_nonpositive = True
    if max_nonpositive:
        max_val = base_max
    try:
        inverted = min_val >= max_val
    except _PARSE_EXCEPTIONS:
        inverted = True
    if inverted:
        min_val, max_val = base_min, base_max

    return {"min": float(min_val), "max": float(max_val)}


def resolve_tiny_relax_from_report(report: dict[str, Any] | None) -> bool:
    """Resolve tiny-relax mode from report context policy fields."""

    if not isinstance(report, dict):
        return False

    ctx = report.get("context")
    if isinstance(ctx, dict):
        run_ctx = ctx.get("run")
        if isinstance(run_ctx, dict):
            run_val = coerce_bool_like(run_ctx.get("tiny_relax"))
            if run_val is not None:
                return bool(run_val)
        eval_ctx = ctx.get("eval")
        if isinstance(eval_ctx, dict):
            eval_val = coerce_bool_like(eval_ctx.get("tiny_relax"))
            if eval_val is not None:
                return bool(eval_val)

    auto = report.get("auto")
    if isinstance(auto, dict):
        auto_val = coerce_bool_like(auto.get("tiny_relax"))
        if auto_val is not None:
            return bool(auto_val)

    provenance = report.get("provenance")
    if isinstance(provenance, dict):
        flags = provenance.get("flags")
        if isinstance(flags, list):
            return "tiny_relax" in {str(flag).strip().lower() for flag in flags}

    return False
