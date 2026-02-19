"""Analysis and normalization helpers for the run command."""

from __future__ import annotations

import math
from typing import Any


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
    import math as _m

    if not isinstance(pm, dict) or not isinstance(metrics, dict):
        return ""
    diffs: list[str] = []
    try:
        pm_blk = metrics.get("primary_metric", {}) if isinstance(metrics, dict) else {}
        ppl_final_v1 = float(pm_blk.get("final", float("nan")))
    except Exception:
        ppl_final_v1 = float("nan")
    try:
        ppl_prev_v1 = float(pm_blk.get("preview", float("nan")))
    except Exception:
        ppl_prev_v1 = float("nan")
    try:
        ppl_final_v2 = float(pm.get("final", float("nan")))
    except Exception:
        ppl_final_v2 = float("nan")
    try:
        ppl_prev_v2 = float(pm.get("preview", float("nan")))
    except Exception:
        ppl_prev_v2 = float("nan")

    if _m.isfinite(ppl_final_v1) and _m.isfinite(ppl_final_v2):
        diffs.append(f"final: v1-v1 = {ppl_final_v2 - ppl_final_v1:+.9f}")
        try:
            diffs.append(
                f"Δlog(final): {_m.log(ppl_final_v2) - _m.log(ppl_final_v1):+.9f}"
            )
        except Exception:
            pass
    if _m.isfinite(ppl_prev_v1) and _m.isfinite(ppl_prev_v2):
        diffs.append(f"preview: v1-v1 = {ppl_prev_v2 - ppl_prev_v1:+.9f}")
        try:
            diffs.append(
                f"Δlog(preview): {_m.log(ppl_prev_v2) - _m.log(ppl_prev_v1):+.9f}"
            )
        except Exception:
            pass

    try:
        r_v2 = float(pm.get("ratio_vs_baseline", float("nan")))
    except Exception:
        r_v2 = float("nan")
    r_v1 = float(pm_blk.get("ratio_vs_baseline", float("nan")))
    if (not _m.isfinite(r_v1)) and isinstance(baseline_report_data, dict):
        try:
            base_fin = float(
                (
                    (baseline_report_data.get("metrics") or {}).get("primary_metric")
                    or {}
                ).get("final")
            )
            if _m.isfinite(base_fin) and base_fin > 0 and _m.isfinite(ppl_final_v1):
                r_v1 = ppl_final_v1 / base_fin
        except Exception:
            pass
    if _m.isfinite(r_v1) and _m.isfinite(r_v2):
        diffs.append(f"ratio_vs_baseline: v1-v1 = {r_v2 - r_v1:+.9f}")
    return "; ".join(diffs)


def normalize_overhead_result(payload: dict[str, object] | None) -> dict[str, object]:
    """Normalize guard-overhead payload for tiny/degenerate runs."""
    payload = dict(payload or {})
    try:
        ratio = payload.get("overhead_ratio")
        val = float(ratio) if isinstance(ratio, int | float) else float("nan")
    except Exception:
        val = float("nan")
    if not (isinstance(val, float) and math.isfinite(val)):
        payload["evaluated"] = False
        payload["passed"] = True
    return payload
