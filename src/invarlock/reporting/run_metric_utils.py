from __future__ import annotations

import math
from typing import Any

_PARSE_EXCEPTIONS = (AttributeError, KeyError, OverflowError, TypeError, ValueError)


def merge_primary_metric_health(
    primary_metric: dict[str, Any] | None,
    core_primary_metric: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge health flags from core primary metric into report primary metric."""
    if not isinstance(primary_metric, dict):
        return {}
    merged = dict(primary_metric)
    if not isinstance(core_primary_metric, dict):
        return merged
    if core_primary_metric.get("invalid") is True:
        merged["invalid"] = True
        merged["degraded"] = True
    if core_primary_metric.get("degraded") is True:
        merged["degraded"] = True
    core_reason = core_primary_metric.get("degraded_reason")
    if isinstance(core_reason, str) and core_reason:
        merged["degraded_reason"] = core_reason
        merged["degraded"] = True
    return merged


def format_debug_metric_diffs(
    pm: dict[str, float] | None,
    metrics: dict[str, float] | None,
    baseline_report_data: dict | None,
) -> str:
    """Build a compact debug line comparing current snapshot vs ppl_* values."""
    if not isinstance(pm, dict) or not isinstance(metrics, dict):
        return ""
    diffs: list[str] = []
    pm_blk: dict[str, Any] = {}
    try:
        raw_pm_blk: Any = metrics.get("primary_metric", {})
        if isinstance(raw_pm_blk, dict):
            pm_blk = raw_pm_blk
        recorded_final = float(pm_blk.get("final", float("nan")))
    except _PARSE_EXCEPTIONS:
        recorded_final = float("nan")
    try:
        recorded_preview = float(pm_blk.get("preview", float("nan")))
    except _PARSE_EXCEPTIONS:
        recorded_preview = float("nan")
    try:
        recomputed_final = float(pm.get("final", float("nan")))
    except _PARSE_EXCEPTIONS:
        recomputed_final = float("nan")
    try:
        recomputed_preview = float(pm.get("preview", float("nan")))
    except _PARSE_EXCEPTIONS:
        recomputed_preview = float("nan")

    if math.isfinite(recorded_final) and math.isfinite(recomputed_final):
        diffs.append(
            f"final: recomputed-recorded = {recomputed_final - recorded_final:+.9f}"
        )
        try:
            diffs.append(
                "Δlog(final): "
                f"{math.log(recomputed_final) - math.log(recorded_final):+.9f}"
            )
        except _PARSE_EXCEPTIONS:
            pass
    if math.isfinite(recorded_preview) and math.isfinite(recomputed_preview):
        diffs.append(
            "preview: recomputed-recorded = "
            f"{recomputed_preview - recorded_preview:+.9f}"
        )
        try:
            diffs.append(
                "Δlog(preview): "
                f"{math.log(recomputed_preview) - math.log(recorded_preview):+.9f}"
            )
        except _PARSE_EXCEPTIONS:
            pass

    try:
        recomputed_ratio = float(pm.get("ratio_vs_baseline", float("nan")))
    except _PARSE_EXCEPTIONS:
        recomputed_ratio = float("nan")
    try:
        recorded_ratio = float(pm_blk.get("ratio_vs_baseline", float("nan")))
    except _PARSE_EXCEPTIONS:
        recorded_ratio = float("nan")
    if (not math.isfinite(recorded_ratio)) and isinstance(baseline_report_data, dict):
        try:
            metrics_block = baseline_report_data.get("metrics") or {}
            primary_metric_block = (
                metrics_block.get("primary_metric", {})
                if isinstance(metrics_block, dict)
                else {}
            )
            base_final_raw = (
                primary_metric_block.get("final")
                if isinstance(primary_metric_block, dict)
                else None
            )
            if base_final_raw is None:
                raise ValueError("missing baseline final metric")
            base_final = float(base_final_raw)
            if (
                math.isfinite(base_final)
                and base_final > 0
                and math.isfinite(recorded_final)
            ):
                recorded_ratio = recorded_final / base_final
        except _PARSE_EXCEPTIONS:
            pass
    if math.isfinite(recorded_ratio) and math.isfinite(recomputed_ratio):
        diffs.append(
            "ratio_vs_baseline: recomputed-recorded = "
            f"{recomputed_ratio - recorded_ratio:+.9f}"
        )
    return "; ".join(diffs)


__all__ = [
    "format_debug_metric_diffs",
    "merge_primary_metric_health",
]
