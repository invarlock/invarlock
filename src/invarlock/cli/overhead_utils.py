from __future__ import annotations

import math
from typing import Any

_OVERHEAD_EXTRACTION_ERRORS = (
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _valid_primary_metric_snapshot(pm: Any) -> dict[str, Any] | None:
    if not isinstance(pm, dict):
        return None
    fin = pm.get("final")
    if isinstance(fin, int | float) and math.isfinite(float(fin)):
        return pm
    return None


def _compute_snapshot_from_report(
    report: dict[str, Any], *, kind: str
) -> dict[str, Any] | None:
    from invarlock.eval.primary_metric import compute_primary_metric_from_report

    computed = compute_primary_metric_from_report(report, kind=kind)
    return _valid_primary_metric_snapshot(computed)


def _extract_pm_snapshot_for_overhead(
    src: object, *, kind: str
) -> dict[str, Any] | None:
    """Extract or compute a primary-metric snapshot from diverse report shapes.

    Accepts either:
    - CoreRunner RunReport-like objects (dataclasses) with `.metrics`/`.evaluation_windows`
    - Dict reports with `evaluation_windows` or `metrics.primary_metric`

    Returns a dict suitable for `metrics.primary_metric` or None if unavailable.
    """
    # 1) Prefer existing primary_metric on object metrics
    try:
        metrics = getattr(src, "metrics", None)
        if isinstance(metrics, dict):
            snapshot = _valid_primary_metric_snapshot(metrics.get("primary_metric"))
            if snapshot is not None:
                return snapshot
    except _OVERHEAD_EXTRACTION_ERRORS:
        pass

    # 2) If dict-shaped report provided, try computing from it directly
    try:
        if isinstance(src, dict):
            snapshot = _compute_snapshot_from_report(src, kind=kind)
            if snapshot is not None:
                return snapshot
    except _OVERHEAD_EXTRACTION_ERRORS:
        pass

    # 3) Compute from evaluation_windows attribute on CoreRunner reports
    try:
        ew = getattr(src, "evaluation_windows", None)
        if isinstance(ew, dict) and ew:
            snapshot = _compute_snapshot_from_report(
                {"evaluation_windows": ew}, kind=kind
            )
            if snapshot is not None:
                return snapshot
    except _OVERHEAD_EXTRACTION_ERRORS:
        pass

    return None


__all__ = ["_extract_pm_snapshot_for_overhead"]
