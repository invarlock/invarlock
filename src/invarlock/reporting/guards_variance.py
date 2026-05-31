from __future__ import annotations

from typing import Any

from .report_types import RunReport

_GUARD_PARSE_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
)


def _find_variance_guard(
    report: RunReport,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    for guard in report.get("guards", []) or []:
        if "variance" not in str(guard.get("name", "")).lower():
            continue
        metrics = guard.get("metrics", {}) or {}
        guard_metrics = metrics if isinstance(metrics, dict) else {}
        policy = guard.get("policy", {}) or {}
        guard_policy = dict(policy) if isinstance(policy, dict) and policy else None
        return guard_metrics, guard_policy
    return {}, None


def _variance_metric_fallback(
    report: RunReport,
    *,
    guard_metrics: dict[str, Any],
    ve_enabled: bool,
    gain: Any,
    ppl_no_ve: Any,
    ppl_with_ve: Any,
) -> tuple[dict[str, Any], bool, Any, Any, Any]:
    if gain is not None:
        return guard_metrics, ve_enabled, gain, ppl_no_ve, ppl_with_ve
    metrics_variance = (report.get("metrics", {}) or {}).get("variance", {})
    if not isinstance(metrics_variance, dict):
        return guard_metrics, ve_enabled, gain, ppl_no_ve, ppl_with_ve
    if not guard_metrics:
        guard_metrics = metrics_variance
    return (
        guard_metrics,
        metrics_variance.get("ve_enabled", ve_enabled),
        metrics_variance.get("gain", gain),
        metrics_variance.get("ppl_no_ve", ppl_no_ve),
        metrics_variance.get("ppl_with_ve", ppl_with_ve),
    )


def _attach_ratio_ci(result: dict[str, Any], ratio_ci: Any) -> None:
    if isinstance(ratio_ci, tuple | list) and len(ratio_ci) == 2:
        try:
            result["ratio_ci"] = (float(ratio_ci[0]), float(ratio_ci[1]))
        except _GUARD_PARSE_EXCEPTIONS:
            pass


def _attach_metadata_fields(
    result: dict[str, Any],
    guard_metrics: dict[str, Any],
) -> None:
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


def _collect_window_ids(provenance: dict[str, Any]) -> list[int]:
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

    _collect(provenance)
    return sorted(window_ids)


def _build_ab_section(guard_metrics: dict[str, Any]) -> dict[str, Any]:
    ab_section: dict[str, Any] = {}
    if guard_metrics.get("ab_seed_used") is not None:
        ab_section["seed"] = guard_metrics["ab_seed_used"]
    if guard_metrics.get("ab_windows_used") is not None:
        ab_section["windows_used"] = guard_metrics["ab_windows_used"]
    if guard_metrics.get("ab_provenance"):
        provenance = guard_metrics["ab_provenance"]
        if isinstance(provenance, dict):
            prov_out = dict(provenance)
            if "window_ids" not in prov_out:
                window_ids = _collect_window_ids(prov_out)
                if window_ids:
                    prov_out["window_ids"] = window_ids
            ab_section["provenance"] = prov_out
        else:
            ab_section["provenance"] = provenance
    if guard_metrics.get("ab_point_estimates"):
        ab_section["point_estimates"] = guard_metrics["ab_point_estimates"]
    return ab_section


def _extract_variance_analysis(report: RunReport) -> dict[str, Any]:
    ve_enabled = False
    gain = None
    ppl_no_ve = None
    ppl_with_ve = None
    ratio_ci: Any = None
    calibration: dict[str, Any] = {}
    guard_metrics, guard_policy = _find_variance_guard(report)
    if guard_metrics:
        ve_enabled = bool(guard_metrics.get("ve_enabled", bool(guard_metrics)))
        gain = guard_metrics.get("ab_gain", guard_metrics.get("gain", None))
        ppl_no_ve = guard_metrics.get("ppl_no_ve", None)
        ppl_with_ve = guard_metrics.get("ppl_with_ve", None)
        ratio_ci = guard_metrics.get("ratio_ci", ratio_ci)
        calibration_candidate = guard_metrics.get("calibration", calibration)
        if isinstance(calibration_candidate, dict):
            calibration = calibration_candidate
    guard_metrics, ve_enabled, gain, ppl_no_ve, ppl_with_ve = _variance_metric_fallback(
        report,
        guard_metrics=guard_metrics,
        ve_enabled=ve_enabled,
        gain=gain,
        ppl_no_ve=ppl_no_ve,
        ppl_with_ve=ppl_with_ve,
    )
    result: dict[str, Any] = {"enabled": ve_enabled, "gain": gain}
    _attach_ratio_ci(result, ratio_ci)
    if calibration:
        result["calibration"] = calibration
    if not ve_enabled and ppl_no_ve is not None and ppl_with_ve is not None:
        result["ppl_no_ve"] = ppl_no_ve
        result["ppl_with_ve"] = ppl_with_ve
    _attach_metadata_fields(result, guard_metrics)
    ab_section = _build_ab_section(guard_metrics)
    if ab_section:
        result["ab_test"] = ab_section
    if guard_policy:
        result["policy"] = guard_policy
    return result
