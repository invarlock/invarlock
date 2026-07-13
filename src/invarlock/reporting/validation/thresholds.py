"""Drift and token-coverage threshold resolution for report validation."""

from __future__ import annotations

import math
from typing import Any

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _coerce_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def resolve_drift_bounds(
    pm_drift_band: dict[str, float] | None,
    *,
    default: tuple[float, float],
) -> tuple[float, float]:
    drift_min, drift_max = default
    if not isinstance(pm_drift_band, dict):
        return drift_min, drift_max
    try:
        candidate_min = _coerce_finite_float(pm_drift_band.get("min"))
        candidate_max = _coerce_finite_float(pm_drift_band.get("max"))
        if (
            candidate_min is not None
            and candidate_max is not None
            and 0 < candidate_min < candidate_max
        ):
            return candidate_min, candidate_max
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass
    return drift_min, drift_max


def _resolve_effective_min_tokens(
    *,
    min_tokens: int,
    pm_policy: dict[str, Any],
    dataset_capacity: dict[str, Any] | None,
) -> int:
    effective = max(0, int(min_tokens))
    try:
        if isinstance(dataset_capacity, dict):
            fraction = float(pm_policy.get("min_token_fraction", 0.0) or 0.0)
            available = _coerce_finite_float(dataset_capacity.get("tokens_available"))
            if available is not None and fraction > 0.0:
                effective = max(effective, int(math.ceil(available * fraction)))
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass
    return effective


def _coverage_meets_floor(metrics: dict[str, Any]) -> bool:
    try:
        coverage = metrics.get("bootstrap", {}).get("coverage")
        if not isinstance(coverage, dict):
            return False
        preview = coverage.get("preview")
        final = coverage.get("final")
        if not (isinstance(preview, dict) and isinstance(final, dict)):
            return False
        preview_used = _coerce_finite_float(preview.get("used"))
        preview_required = _coerce_finite_float(preview.get("required"))
        final_used = _coerce_finite_float(final.get("used"))
        final_required = _coerce_finite_float(final.get("required"))
        preview_ok = bool(preview.get("ok")) or (
            preview_used is not None
            and preview_required is not None
            and preview_used >= preview_required
        )
        final_ok = bool(final.get("ok")) or (
            final_used is not None
            and final_required is not None
            and final_used >= final_required
        )
        return preview_ok and final_ok
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        return False


def resolve_tokens_ok(
    metrics: dict[str, Any] | None,
    *,
    min_tokens: int,
    pm_policy: dict[str, Any],
    dataset_capacity: dict[str, Any] | None,
) -> bool:
    if not isinstance(metrics, dict):
        return True
    preview_tokens = _coerce_finite_float(metrics.get("preview_total_tokens"))
    final_tokens = _coerce_finite_float(metrics.get("final_total_tokens"))
    if preview_tokens is None or final_tokens is None or min_tokens <= 0:
        return True
    try:
        total_tokens = int(preview_tokens) + int(final_tokens)
        effective_minimum = _resolve_effective_min_tokens(
            min_tokens=min_tokens,
            pm_policy=pm_policy,
            dataset_capacity=dataset_capacity,
        )
        tokens_ok = total_tokens >= effective_minimum
        if tokens_ok or not _coverage_meets_floor(metrics):
            return tokens_ok
        try:
            tolerance = float(pm_policy.get("min_tokens_tolerance", 0.02) or 0.0)
        except _NON_FATAL_EXCEPTIONS:
            tolerance = 0.0
        tolerance = max(tolerance, 0.0)
        relaxed_floor = int(math.floor(effective_minimum * (1.0 - tolerance)))
        return total_tokens >= max(relaxed_floor, 0)
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        return True


__all__ = ["resolve_drift_bounds", "resolve_tokens_ok"]
