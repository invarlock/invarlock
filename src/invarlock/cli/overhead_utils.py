from __future__ import annotations

import math
from typing import Any


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
            pm = metrics.get("primary_metric")
            if isinstance(pm, dict):
                fin = pm.get("final")
                if isinstance(fin, int | float) and math.isfinite(float(fin)):
                    return pm  # already a valid snapshot
    except Exception:
        pass

    # 2) If dict-shaped report provided, try computing from it directly
    try:
        if isinstance(src, dict):
            from invarlock.eval.primary_metric import compute_primary_metric_from_report

            pm2 = compute_primary_metric_from_report(src, kind=kind)
            fin2 = pm2.get("final") if isinstance(pm2, dict) else None
            if isinstance(fin2, int | float) and math.isfinite(float(fin2)):
                return pm2
    except Exception:
        pass

    # 3) Compute from evaluation_windows attribute on CoreRunner reports
    try:
        ew = getattr(src, "evaluation_windows", None)
        if isinstance(ew, dict) and ew:
            from invarlock.eval.primary_metric import compute_primary_metric_from_report

            pm3 = compute_primary_metric_from_report(
                {"evaluation_windows": ew}, kind=kind
            )
            fin3 = pm3.get("final") if isinstance(pm3, dict) else None
            if isinstance(fin3, int | float) and math.isfinite(float(fin3)):
                return pm3
    except Exception:
        pass

    return None


__all__ = ["_extract_pm_snapshot_for_overhead"]
