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
        ppl_final_v1 = float(pm_blk.get("final", float("nan")))
    except _PARSE_EXCEPTIONS:
        ppl_final_v1 = float("nan")
    try:
        ppl_prev_v1 = float(pm_blk.get("preview", float("nan")))
    except _PARSE_EXCEPTIONS:
        ppl_prev_v1 = float("nan")
    try:
        ppl_final_v2 = float(pm.get("final", float("nan")))
    except _PARSE_EXCEPTIONS:
        ppl_final_v2 = float("nan")
    try:
        ppl_prev_v2 = float(pm.get("preview", float("nan")))
    except _PARSE_EXCEPTIONS:
        ppl_prev_v2 = float("nan")

    if math.isfinite(ppl_final_v1) and math.isfinite(ppl_final_v2):
        diffs.append(f"final: v1-v1 = {ppl_final_v2 - ppl_final_v1:+.9f}")
        try:
            diffs.append(
                f"Δlog(final): {math.log(ppl_final_v2) - math.log(ppl_final_v1):+.9f}"
            )
        except _PARSE_EXCEPTIONS:
            pass
    if math.isfinite(ppl_prev_v1) and math.isfinite(ppl_prev_v2):
        diffs.append(f"preview: v1-v1 = {ppl_prev_v2 - ppl_prev_v1:+.9f}")
        try:
            diffs.append(
                f"Δlog(preview): {math.log(ppl_prev_v2) - math.log(ppl_prev_v1):+.9f}"
            )
        except _PARSE_EXCEPTIONS:
            pass

    try:
        ratio_v2 = float(pm.get("ratio_vs_baseline", float("nan")))
    except _PARSE_EXCEPTIONS:
        ratio_v2 = float("nan")
    try:
        ratio_v1 = float(pm_blk.get("ratio_vs_baseline", float("nan")))
    except _PARSE_EXCEPTIONS:
        ratio_v1 = float("nan")
    if (not math.isfinite(ratio_v1)) and isinstance(baseline_report_data, dict):
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
                and math.isfinite(ppl_final_v1)
            ):
                ratio_v1 = ppl_final_v1 / base_final
        except _PARSE_EXCEPTIONS:
            pass
    if math.isfinite(ratio_v1) and math.isfinite(ratio_v2):
        diffs.append(f"ratio_vs_baseline: v1-v1 = {ratio_v2 - ratio_v1:+.9f}")
    return "; ".join(diffs)
