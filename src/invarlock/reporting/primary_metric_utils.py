from __future__ import annotations

import copy
import math
from typing import Any

from .utils import _coerce_interval, _weighted_mean

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
        coerced = float(value)
    except _NON_FATAL_EXCEPTIONS:
        return None
    return coerced if math.isfinite(coerced) else None


def _is_non_bool_finite_number(value: Any) -> bool:
    return _coerce_finite_float(value) is not None


def _resolve_degraded_reason(
    pm_copy: dict[str, Any],
    *,
    preview_value: float | None,
    final_value: float | None,
    ratio_value: float | None,
    baseline_final_value: float | None,
) -> str | None:
    degraded_reason: str | None = pm_copy.get("degraded_reason")
    baseline_has_reference = baseline_final_value is not None
    needs_pm_fallback = not (preview_value is not None and final_value is not None)
    needs_ratio_fallback = baseline_has_reference and ratio_value is None
    can_recompute_ratio = (
        final_value is not None
        and baseline_final_value is not None
        and baseline_final_value > 0.0
    )
    if degraded_reason is None:
        if needs_pm_fallback:
            degraded_reason = "non_finite_pm"
        elif needs_ratio_fallback and not can_recompute_ratio:
            degraded_reason = "non_finite_delta"
        elif pm_copy.get("invalid"):
            degraded_reason = "primary_metric_invalid"
    return degraded_reason


def _resolve_logspace_ci(
    metrics_map: dict[str, Any],
    ppl_analysis: dict[str, Any] | None,
) -> tuple[float, float] | list[float] | None:
    try:
        dlci_source: tuple[float, float] | list[float] | None = None
        pairing_source = None
        if isinstance(ppl_analysis, dict):
            stats = ppl_analysis.get("stats") or {}
            if isinstance(stats, dict):
                pairing_source = stats.get("pairing")
            if pairing_source == "paired_baseline":
                dlci_source = _coerce_interval(ppl_analysis.get("logloss_delta_ci"))
        if dlci_source is None:
            dlci_source = (
                _coerce_interval(metrics_map.get("logloss_delta_ci"))
                if isinstance(metrics_map, dict)
                else (math.nan, math.nan)
            )
        return dlci_source
    except _NON_FATAL_EXCEPTIONS:
        return None


def _attach_ppl_analysis_fields(
    pm_copy: dict[str, Any],
    *,
    report: dict[str, Any],
    metrics_map: dict[str, Any],
    ppl_analysis: dict[str, Any] | None,
) -> None:
    try:
        kind = str(pm_copy.get("kind", "")).lower()
    except _NON_FATAL_EXCEPTIONS:
        kind = ""
    if not kind.startswith("ppl"):
        return
    try:
        eval_windows = (
            report.get("evaluation_windows", {})
            if isinstance(report.get("evaluation_windows"), dict)
            else {}
        )
        prev_sec = (
            eval_windows.get("preview") if isinstance(eval_windows, dict) else None
        )
        fin_sec = eval_windows.get("final") if isinstance(eval_windows, dict) else None
        if isinstance(prev_sec, dict) and isinstance(fin_sec, dict):
            mean_prev = _weighted_mean(
                list(prev_sec.get("logloss", []) or []),
                list(prev_sec.get("token_counts", []) or []),
            )
            mean_fin = _weighted_mean(
                list(fin_sec.get("logloss", []) or []),
                list(fin_sec.get("token_counts", []) or []),
            )
            if math.isfinite(mean_prev):
                pm_copy["analysis_basis"] = "mean_logloss"
                pm_copy["analysis_point_preview"] = float(mean_prev)
            if math.isfinite(mean_fin):
                pm_copy["analysis_basis"] = "mean_logloss"
                pm_copy["analysis_point_final"] = float(mean_fin)
        dlci_source = _resolve_logspace_ci(metrics_map, ppl_analysis)
        if isinstance(dlci_source, (tuple, list)) and len(dlci_source) == 2:
            lo_raw, hi_raw = dlci_source[0], dlci_source[1]
            if _is_non_bool_finite_number(lo_raw) and _is_non_bool_finite_number(
                hi_raw
            ):
                pm_copy.setdefault("ci", [float(lo_raw), float(hi_raw)])
    except _NON_FATAL_EXCEPTIONS:
        pass


def _finalize_primary_metric_snapshot(
    pm_copy: dict[str, Any],
    *,
    report: dict[str, Any],
    metrics_map: dict[str, Any],
    baseline_ref: dict[str, Any] | None,
    ppl_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    pm_copy.setdefault("invalid", bool(pm_copy.get("invalid", False)))
    preview_value = _coerce_finite_float(pm_copy.get("preview"))
    final_value = _coerce_finite_float(pm_copy.get("final"))
    ratio_value = _coerce_finite_float(pm_copy.get("ratio_vs_baseline"))
    baseline_final = (
        baseline_ref.get("primary_metric", {}).get("final")
        if isinstance(baseline_ref, dict)
        else None
    )
    baseline_final_value = _coerce_finite_float(baseline_final)
    degraded_reason = _resolve_degraded_reason(
        pm_copy,
        preview_value=preview_value,
        final_value=final_value,
        ratio_value=ratio_value,
        baseline_final_value=baseline_final_value,
    )
    pm_copy["degraded"] = bool(
        pm_copy.get("degraded") or pm_copy.get("invalid") or degraded_reason
    )
    if pm_copy["degraded"] and degraded_reason:
        pm_copy.setdefault("degraded_reason", degraded_reason)
    try:
        if isinstance(ppl_analysis, dict) and bool(ppl_analysis.get("unstable")):
            pm_copy.setdefault("unstable", True)
    except _NON_FATAL_EXCEPTIONS:
        pass
    _attach_ppl_analysis_fields(
        pm_copy,
        report=report,
        metrics_map=metrics_map,
        ppl_analysis=ppl_analysis,
    )
    try:
        if (
            final_value is not None
            and baseline_final_value is not None
            and baseline_final_value > 0
        ):
            pm_copy["ratio_vs_baseline"] = final_value / baseline_final_value
        try:
            kind = str(pm_copy.get("kind", "")).lower()
        except _NON_FATAL_EXCEPTIONS:
            kind = ""
        ci = pm_copy.get("ci")
        if kind.startswith("ppl") and isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                lo, hi = float(ci[0]), float(ci[1])
                if math.isfinite(lo) and math.isfinite(hi):
                    pm_copy["display_ci"] = [math.exp(lo), math.exp(hi)]
            except _NON_FATAL_EXCEPTIONS:
                pass
        if (
            not isinstance(pm_copy.get("display_ci"), (list, tuple))
            and final_value is not None
        ):
            pm_copy["display_ci"] = [final_value, final_value]
    except _NON_FATAL_EXCEPTIONS:
        pass
    return pm_copy


def _attach_primary_metric_from_report(
    evaluation_report: dict[str, Any],
    report: dict[str, Any],
    baseline_ref: dict[str, Any] | None,
    ppl_analysis: dict[str, Any] | None,
) -> None:
    try:
        metrics_map = (
            report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
        )
        pm = (
            metrics_map.get("primary_metric") if isinstance(metrics_map, dict) else None
        )
        if isinstance(pm, dict) and pm:
            evaluation_report["primary_metric"] = _finalize_primary_metric_snapshot(
                copy.deepcopy(pm),
                report=report,
                metrics_map=metrics_map,
                baseline_ref=baseline_ref,
                ppl_analysis=ppl_analysis,
            )
    except _NON_FATAL_EXCEPTIONS:
        pass


def _attach_primary_metric_from_windows(
    evaluation_report: dict[str, Any],
    report: dict[str, Any],
    baseline_raw: dict[str, Any] | None,
) -> None:
    if isinstance(evaluation_report.get("primary_metric"), dict):
        return
    metrics_map = (
        report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    )
    loss_type = (
        (metrics_map.get("loss_type") or "").lower()
        if isinstance(metrics_map, dict)
        else ""
    )
    if loss_type == "mlm":
        kind_hint = "ppl_mlm"
    elif loss_type in {"seq2seq", "s2s", "t5"}:
        kind_hint = "ppl_seq2seq"
    else:
        kind_hint = "ppl_causal"
    from invarlock.eval.primary_metric import (
        compute_primary_metric_from_report as _pm,
    )

    for _attempt in range(2):
        try:
            pm_block = _pm(
                report,
                kind=kind_hint,
                baseline=baseline_raw if isinstance(baseline_raw, dict) else None,
            )
        except _NON_FATAL_EXCEPTIONS:
            continue
        if isinstance(pm_block, dict) and pm_block:
            evaluation_report["primary_metric"] = pm_block
            return


def _attach_classification_primary_metric_fallback(
    evaluation_report: dict[str, Any],
    report: dict[str, Any],
    baseline_raw: dict[str, Any] | None,
    baseline_ref: dict[str, Any] | None,
) -> None:
    if isinstance(evaluation_report.get("primary_metric"), dict):
        return
    try:
        metrics_map = report.get("metrics", {}) if isinstance(report, dict) else {}
        clf = (
            metrics_map.get("classification") if isinstance(metrics_map, dict) else None
        )
        if not (isinstance(clf, dict) and clf):
            return
        pm_kind = "accuracy"
        pm_point = None
        try:
            val_value = _coerce_finite_float(clf.get("final"))
            if val_value is not None:
                pm_point = val_value
            else:
                val = clf.get("final")
                if not isinstance(val, dict):
                    raise TypeError(
                        "classification.final must be a dict when not scalar"
                    )
                num_value = _coerce_finite_float(val.get("correct_total"))
                den_value = _coerce_finite_float(val.get("total"))
                if num_value is not None and den_value is not None and den_value > 0:
                    pm_point = num_value / den_value
        except _NON_FATAL_EXCEPTIONS:
            pm_point = None
        acc_pm: dict[str, Any] = {
            "kind": pm_kind,
            "unit": "accuracy",
            "direction": "higher",
            "aggregation_scope": "example",
            "paired": True,
            "gating_basis": "point",
        }
        if isinstance(pm_point, float):
            acc_pm["final"] = pm_point
            acc_pm.setdefault("display_ci", [pm_point, pm_point])
        try:
            base_cls = None
            if isinstance(baseline_raw, dict):
                bm = (
                    baseline_raw.get("metrics")
                    if isinstance(baseline_raw.get("metrics"), dict)
                    else None
                )
                if isinstance(bm, dict):
                    base_cls = bm.get("classification")
            if base_cls is None and isinstance(baseline_ref, dict):
                bm = (
                    baseline_ref.get("metrics")
                    if isinstance(baseline_ref.get("metrics"), dict)
                    else None
                )
                if isinstance(bm, dict):
                    base_cls = bm.get("classification")
            acc_base = None
            if isinstance(base_cls, dict):
                valb_value = _coerce_finite_float(base_cls.get("final"))
                if valb_value is not None:
                    acc_base = valb_value
                else:
                    valb = base_cls.get("final")
                    if not isinstance(valb, dict):
                        raise TypeError(
                            "baseline classification.final must be a dict when not scalar"
                        )
                    nb_value = _coerce_finite_float(valb.get("correct_total"))
                    db_value = _coerce_finite_float(valb.get("total"))
                    if nb_value is not None and db_value is not None and db_value > 0:
                        acc_base = nb_value / db_value
            if isinstance(pm_point, float) and isinstance(acc_base, float):
                acc_pm["ratio_vs_baseline"] = (pm_point - acc_base) * 100.0
        except _NON_FATAL_EXCEPTIONS:
            pass
        evaluation_report["primary_metric"] = acc_pm
    except _NON_FATAL_EXCEPTIONS:
        pass


def _ensure_primary_metric_display_ci(evaluation_report: dict[str, Any]) -> None:
    try:
        pm = (
            evaluation_report.get("primary_metric", {})
            if isinstance(evaluation_report.get("primary_metric"), dict)
            else None
        )
        if not (isinstance(pm, dict) and pm):
            return
        disp = pm.get("display_ci")
        if (
            isinstance(disp, (list, tuple))
            and len(disp) == 2
            and all(_is_non_bool_finite_number(x) for x in disp)
        ):
            return
        point = None
        for key in ("ratio_vs_baseline", "final", "preview"):
            point_value = _coerce_finite_float(pm.get(key))
            if point_value is not None:
                point = point_value
                break
        if isinstance(point, float):
            pm["display_ci"] = [point, point]
        else:
            pm["display_ci"] = [1.0, 1.0]
            pm.setdefault("estimated", True)
    except _NON_FATAL_EXCEPTIONS:
        pass


def attach_primary_metric(
    evaluation_report: dict[str, Any],
    report: dict[str, Any],
    baseline_raw: dict[str, Any] | None,
    baseline_ref: dict[str, Any] | None,
    ppl_analysis: dict[str, Any] | None,
) -> None:
    """Attach/normalize the primary_metric block on the evaluation report.

    Behavior matches the canonical evaluation-report assembly contract and preserves structure:
    - Prefer explicit metrics.primary_metric if present
    - Compute missing ratio_vs_baseline, degenerate display_ci
    - ppl window-based analysis info (mean logloss) added when available
    - Fallbacks for classification metrics and eval-window-derived ppl
    - Ensure display_ci always present for schema invariants
    Mutates the evaluation report in-place.
    """
    _attach_primary_metric_from_report(
        evaluation_report,
        report,
        baseline_ref,
        ppl_analysis,
    )
    _attach_primary_metric_from_windows(evaluation_report, report, baseline_raw)
    _attach_classification_primary_metric_fallback(
        evaluation_report,
        report,
        baseline_raw,
        baseline_ref,
    )
    _attach_primary_metric_from_windows(evaluation_report, report, baseline_raw)
    _ensure_primary_metric_display_ci(evaluation_report)
