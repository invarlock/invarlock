from __future__ import annotations

import math
from typing import Any

_NON_FATAL_EXCEPTIONS = (AttributeError, RuntimeError, TypeError, ValueError)


def compute_confidence_label(evaluation_report: dict[str, Any]) -> dict[str, Any]:
    validation = evaluation_report.get("validation", {}) or {}
    pm_ok = bool(validation.get("primary_metric_acceptable", False))
    basis = "primary_metric"
    lo = hi = float("nan")
    try:
        primary_metric = evaluation_report.get("primary_metric", {}) or {}
        kind = str(primary_metric.get("kind", "") or "").lower()
        display_ci = primary_metric.get("display_ci")
        if isinstance(display_ci, tuple | list) and len(display_ci) == 2:
            lo, hi = float(display_ci[0]), float(display_ci[1])
            if kind.startswith("ppl"):
                basis = "ppl_ratio"
            elif kind == "accuracy":
                basis = kind
    except _NON_FATAL_EXCEPTIONS:
        pass

    width = hi - lo if math.isfinite(lo) and math.isfinite(hi) else float("nan")
    ratio_threshold = 0.03
    accuracy_threshold = 1.0
    try:
        resolved_policy = evaluation_report.get("resolved_policy")
        if isinstance(resolved_policy, dict):
            confidence_policy = resolved_policy.get("confidence")
            if isinstance(confidence_policy, dict):
                ppl_ratio_width = confidence_policy.get("ppl_ratio_width_max")
                if isinstance(ppl_ratio_width, int | float):
                    ratio_threshold = float(ppl_ratio_width)
                accuracy_delta = confidence_policy.get("accuracy_delta_pp_width_max")
                if isinstance(accuracy_delta, int | float):
                    accuracy_threshold = float(accuracy_delta)
    except _NON_FATAL_EXCEPTIONS:
        pass
    threshold = accuracy_threshold if basis == "accuracy" else ratio_threshold

    try:
        unstable = bool((evaluation_report.get("primary_metric") or {}).get("unstable"))
    except _NON_FATAL_EXCEPTIONS:
        unstable = False

    label = "Low"
    if pm_ok:
        if (not unstable) and math.isfinite(width) and width <= threshold:
            label = "High"
        elif math.isfinite(width) and width <= 2 * threshold:
            label = "Medium"
        else:
            label = "Medium" if unstable else "Low"

    return {
        "label": label,
        "basis": basis,
        "width": width,
        "threshold": threshold,
        "unstable": unstable,
    }
