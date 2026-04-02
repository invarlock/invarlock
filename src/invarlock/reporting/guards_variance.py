from __future__ import annotations

from typing import Any

from .report_types import RunReport

_PARSE_EXCEPTIONS = (AttributeError, KeyError, OverflowError, TypeError, ValueError)


def _extract_variance_analysis(report: RunReport) -> dict[str, Any]:
    ve_enabled = False
    gain = None
    ppl_no_ve = None
    ppl_with_ve = None
    ratio_ci: Any = None
    calibration: dict[str, Any] = {}
    guard_metrics: dict[str, Any] = {}
    guard_policy: dict[str, Any] | None = None
    for guard in report.get("guards", []) or []:
        if "variance" in str(guard.get("name", "")).lower():
            metrics = guard.get("metrics", {}) or {}
            guard_metrics = metrics if isinstance(metrics, dict) else {}
            gp = guard.get("policy", {}) or {}
            if isinstance(gp, dict) and gp:
                guard_policy = dict(gp)
            ve_enabled = bool(guard_metrics.get("ve_enabled", bool(guard_metrics)))
            gain = guard_metrics.get("ab_gain", guard_metrics.get("gain", None))
            ppl_no_ve = guard_metrics.get("ppl_no_ve", None)
            ppl_with_ve = guard_metrics.get("ppl_with_ve", None)
            ratio_ci = guard_metrics.get("ratio_ci", ratio_ci)
            calibration_candidate = guard_metrics.get("calibration", calibration)
            if isinstance(calibration_candidate, dict):
                calibration = calibration_candidate
            break
    if gain is None:
        metrics_variance = (report.get("metrics", {}) or {}).get("variance", {})
        if isinstance(metrics_variance, dict):
            ve_enabled = metrics_variance.get("ve_enabled", ve_enabled)
            gain = metrics_variance.get("gain", gain)
            ppl_no_ve = metrics_variance.get("ppl_no_ve", ppl_no_ve)
            ppl_with_ve = metrics_variance.get("ppl_with_ve", ppl_with_ve)
            if not guard_metrics:
                guard_metrics = metrics_variance
    result: dict[str, Any] = {"enabled": ve_enabled, "gain": gain}
    if isinstance(ratio_ci, tuple | list) and len(ratio_ci) == 2:
        try:
            result["ratio_ci"] = (float(ratio_ci[0]), float(ratio_ci[1]))
        except _PARSE_EXCEPTIONS:
            pass
    if calibration:
        result["calibration"] = calibration
    if not ve_enabled and ppl_no_ve is not None and ppl_with_ve is not None:
        result["ppl_no_ve"] = ppl_no_ve
        result["ppl_with_ve"] = ppl_with_ve
    metadata_fields = [
        "tap",
        "target_modules",
        "target_module_names",
        "focus_modules",
        "scope",
        "proposed_scales",
        "proposed_scales_pre_edit",
        "proposed_scales_post_edit",
        "monitor_only",
        "max_calib_used",
        "mode",
        "min_rel_gain",
        "alpha",
    ]
    for field in metadata_fields:
        value = guard_metrics.get(field)
        if value not in (None, {}, []):
            result[field] = value
    predictive_gate = guard_metrics.get("predictive_gate")
    if predictive_gate:
        result["predictive_gate"] = predictive_gate
    ab_section: dict[str, Any] = {}
    if guard_metrics.get("ab_seed_used") is not None:
        ab_section["seed"] = guard_metrics["ab_seed_used"]
    if guard_metrics.get("ab_windows_used") is not None:
        ab_section["windows_used"] = guard_metrics["ab_windows_used"]
    if guard_metrics.get("ab_provenance"):
        prov = guard_metrics["ab_provenance"]
        if isinstance(prov, dict):
            prov_out = dict(prov)

            if "window_ids" not in prov_out:
                window_ids: set[int] = set()

                def _collect(node: Any) -> None:
                    if isinstance(node, dict):
                        ids = node.get("window_ids")
                        if isinstance(ids, list):
                            for wid in ids:
                                if isinstance(wid, int | float):
                                    window_ids.add(int(wid))
                        for value in node.values():
                            _collect(value)
                        return
                    if isinstance(node, list):
                        for value in node:
                            _collect(value)

                _collect(prov_out)
                if window_ids:
                    prov_out["window_ids"] = sorted(window_ids)

            ab_section["provenance"] = prov_out
        else:
            ab_section["provenance"] = prov
    if guard_metrics.get("ab_point_estimates"):
        ab_section["point_estimates"] = guard_metrics["ab_point_estimates"]
    if ab_section:
        result["ab_test"] = ab_section
    if guard_policy:
        result["policy"] = guard_policy
    return result
