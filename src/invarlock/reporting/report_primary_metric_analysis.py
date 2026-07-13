from __future__ import annotations

import copy
import math
from typing import Any

from invarlock.core import bootstrap as bootstrap_mod
from invarlock.core.auto_tuning import get_tier_policies
from invarlock.eval.primary_metric import compute_primary_metric_from_report

from .report_primary_metric_counts import _as_count as _as_count
from .report_primary_metric_counts import (
    _classification_total as _classification_total,
)
from .report_primary_metric_counts import _count_examples as _count_examples
from .report_primary_metric_counts import (
    _populate_stats_with_counts_and_coverage as _populate_stats_with_counts_and_coverage,
)
from .report_primary_metric_policy import (
    enforce_drift_ratio_identity,
    enforce_pairing_and_coverage,
    enforce_ratio_ci_alignment,
)
from .utils import _coerce_interval, _pair_logloss_windows

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    RuntimeError,
    OSError,
)
_NUMERIC_EXCEPTIONS = (TypeError, ValueError, OverflowError)


def _report_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report, dict) else None
    return metrics if isinstance(metrics, dict) else {}


def _resolve_auto_tier(report: dict[str, Any]) -> str:
    auto_tier = "balanced"
    try:
        auto_cfg = (
            report.get("meta", {}).get("auto") if isinstance(report, dict) else None
        )
        if isinstance(auto_cfg, dict) and auto_cfg.get("tier"):
            auto_tier = str(auto_cfg.get("tier")).lower()
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        auto_tier = "balanced"
    return auto_tier


def _collect_bootstrap_context(
    report: dict[str, Any],
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None, str | None
]:
    metrics = _report_metrics(report)
    metrics_bootstrap_obj = metrics.get("bootstrap", {})
    metrics_bootstrap = (
        dict(metrics_bootstrap_obj) if isinstance(metrics_bootstrap_obj, dict) else {}
    )
    raw_coverage = metrics_bootstrap.get("coverage") if metrics_bootstrap else None
    coverage_summary = (
        copy.deepcopy(raw_coverage) if isinstance(raw_coverage, dict) else {}
    )
    window_plan_ctx = metrics.get("window_plan")
    window_plan_profile = (
        str(window_plan_ctx.get("profile"))
        if isinstance(window_plan_ctx, dict) and window_plan_ctx.get("profile")
        else None
    )
    return (
        metrics,
        metrics_bootstrap,
        coverage_summary,
        window_plan_ctx,
        window_plan_profile,
    )


def _resolve_ratio_ci_from_run_metrics(
    metrics: dict[str, Any],
) -> tuple[tuple[float, float] | None, str]:
    ratio_ci: tuple[float, float] | None = None
    if isinstance(metrics.get("preview_final_slice_delta_summary"), dict):
        # The runtime interval compares two disjoint subject slices. It is drift
        # evidence and must not be promoted to a baseline ratio interval.
        return None, "independent_preview_final"
    ratio_ci_source = "run_metrics"
    try:
        dlci = _coerce_interval(metrics.get("logloss_delta_ci"))
        if (
            isinstance(dlci, tuple | list)
            and len(dlci) == 2
            and all(isinstance(x, (int | float)) for x in dlci)
        ):
            lo, hi = float(dlci[0]), float(dlci[1])
            ratio_ci = (math.exp(lo), math.exp(hi))
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass
    return ratio_ci, ratio_ci_source


def _resolve_unstable_ci_flag(
    report: dict[str, Any],
    metrics: dict[str, Any],
    metrics_bootstrap: dict[str, Any],
) -> bool:
    unstable_ci_flag = False
    try:
        rep_raw = metrics_bootstrap.get("replicates", metrics_bootstrap.get("n"))
        if rep_raw is not None and int(rep_raw) < 200:
            unstable_ci_flag = True
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        unstable_ci_flag = False

    try:
        tokens_prev = metrics.get("preview_total_tokens")
        tokens_fin = metrics.get("final_total_tokens")
        total_tokens = None
        if isinstance(tokens_prev, int | float) and isinstance(tokens_fin, int | float):
            total_tokens = int(tokens_prev) + int(tokens_fin)

        tier = _resolve_auto_tier(report)
        tier_policies = get_tier_policies()
        tier_defaults = tier_policies.get(tier, tier_policies.get("balanced", {}))
        metrics_policy = (
            tier_defaults.get("metrics", {}) if isinstance(tier_defaults, dict) else {}
        )
        pm_policy = (
            metrics_policy.get("pm_ratio", {})
            if isinstance(metrics_policy, dict)
            else {}
        )
        min_tokens = int(pm_policy.get("min_tokens", 0))
        if (
            isinstance(total_tokens, int)
            and min_tokens > 0
            and total_tokens < min_tokens
        ):
            unstable_ci_flag = True
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    return unstable_ci_flag


def _resolve_paired_weights(
    run_windows: dict[str, Any],
    baseline_windows: dict[str, Any],
) -> list[float] | None:
    paired_weights: list[float] | None = None
    try:
        run_ids = (
            run_windows.get("window_ids") if isinstance(run_windows, dict) else None
        )
        run_w = (
            run_windows.get("token_counts") if isinstance(run_windows, dict) else None
        )
        base_ids = (
            baseline_windows.get("window_ids")
            if isinstance(baseline_windows, dict)
            else None
        )
        if (
            isinstance(run_ids, list)
            and isinstance(run_w, list)
            and isinstance(base_ids, list)
        ):
            base_set = {int(b_id) for b_id in base_ids if isinstance(b_id, int | float)}
            weights: list[float] = []
            for r_id, w in zip(run_ids, run_w, strict=False):
                if not isinstance(r_id, int | float):
                    continue
                key = int(r_id)
                if key not in base_set:
                    continue
                try:
                    wv = float(w)
                except _NUMERIC_EXCEPTIONS:
                    continue
                if not math.isfinite(wv):
                    continue
                weights.append(float(max(wv, 0.0)))
            if weights:
                paired_weights = weights
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        paired_weights = None
    return paired_weights


def _resolve_baseline_delta_mean(
    run_windows: dict[str, Any],
    baseline_windows: dict[str, Any],
) -> float:
    baseline_delta_mean = float("nan")
    try:
        run_ids = (
            run_windows.get("window_ids") if isinstance(run_windows, dict) else None
        )
        run_ll = run_windows.get("logloss") if isinstance(run_windows, dict) else None
        base_ids = (
            baseline_windows.get("window_ids")
            if isinstance(baseline_windows, dict)
            else None
        )
        base_ll = (
            baseline_windows.get("logloss")
            if isinstance(baseline_windows, dict)
            else None
        )
        run_w = (
            run_windows.get("token_counts") if isinstance(run_windows, dict) else None
        )
        if (
            isinstance(run_ids, list)
            and isinstance(run_ll, list)
            and isinstance(base_ids, list)
            and isinstance(base_ll, list)
            and isinstance(run_w, list)
        ):
            base_map: dict[int, float] = {}
            for b_id, b_val in zip(base_ids, base_ll, strict=False):
                if isinstance(b_id, int | float) and isinstance(b_val, int | float):
                    base_map[int(b_id)] = float(b_val)
            sum_w = 0.0
            sum_dw = 0.0
            for r_id, r_val, w in zip(run_ids, run_ll, run_w, strict=False):
                if not (
                    isinstance(r_id, int | float) and isinstance(r_val, int | float)
                ):
                    continue
                try:
                    wv = float(w)
                except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
                    continue
                if not math.isfinite(wv) or wv <= 0:
                    continue
                key = int(r_id)
                if key not in base_map:
                    continue
                sum_w += wv
                sum_dw += wv * (float(r_val) - base_map[key])
            if sum_w > 0.0:
                baseline_delta_mean = float(sum_dw / sum_w)
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        baseline_delta_mean = float("nan")
    return baseline_delta_mean


def _resolve_paired_window_analysis(
    report: dict[str, Any],
    baseline_normalized: dict[str, Any],
    metrics_bootstrap: dict[str, Any],
    logloss_delta_ci: tuple[float, float] | None,
    ratio_ci: tuple[float, float] | None,
    ratio_ci_source: str,
) -> tuple[int, float, tuple[float, float] | None, tuple[float, float] | None, str]:
    run_windows = (
        report.get("evaluation_windows", {}).get("final", {})
        if isinstance(report.get("evaluation_windows"), dict)
        else {}
    )
    baseline_windows = (
        baseline_normalized.get("evaluation_windows", {}).get("final", {})
        if isinstance(baseline_normalized, dict)
        else {}
    )

    paired = _pair_logloss_windows(run_windows, baseline_windows)
    paired_windows = 0
    baseline_delta_mean = float("nan")
    if not paired:
        return (
            paired_windows,
            baseline_delta_mean,
            logloss_delta_ci,
            ratio_ci,
            ratio_ci_source,
        )

    paired_run, paired_base = paired
    paired_windows = len(paired_run)
    paired_weights = _resolve_paired_weights(run_windows, baseline_windows)

    method = str(metrics_bootstrap.get("method", "percentile")).lower()
    replicates = int(
        metrics_bootstrap.get("replicates", metrics_bootstrap.get("n", 1000) or 1000)
    )
    alpha = float(metrics_bootstrap.get("alpha", 0.05) or 0.05)
    seed = int(metrics_bootstrap.get("seed", 0) or 0)
    ci_method = "percentile"
    if "bca" in method:
        ci_method = "bca"
    if replicates <= 0:
        return (
            paired_windows,
            baseline_delta_mean,
            logloss_delta_ci,
            ratio_ci,
            ratio_ci_source,
        )

    try:
        delta_ci = bootstrap_mod.compute_paired_delta_log_ci(
            paired_run,
            paired_base,
            weights=paired_weights,
            method=ci_method,
            replicates=replicates,
            alpha=alpha,
            seed=seed + bootstrap_mod.PAIRED_BASELINE_BOOTSTRAP_SEED_OFFSET,
        )
        if isinstance(delta_ci, tuple | list) and len(delta_ci) == 2:
            delta_ci = (float(delta_ci[0]), float(delta_ci[1]))
        logloss_delta_ci = delta_ci
        ratio_ci = bootstrap_mod.logspace_to_ratio_ci(delta_ci)
        ratio_ci_source = "paired_baseline"
        baseline_delta_mean = _resolve_baseline_delta_mean(
            run_windows, baseline_windows
        )
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        ratio_ci_source = "run_metrics"

    return (
        paired_windows,
        baseline_delta_mean,
        logloss_delta_ci,
        ratio_ci,
        ratio_ci_source,
    )


def _identical_final_id_pair_count(
    report: dict[str, Any],
    baseline_normalized: dict[str, Any],
) -> int:
    """Count exact baseline/subject final-ID pairs for non-logloss metrics."""

    report_windows = report.get("evaluation_windows")
    baseline_windows = baseline_normalized.get("evaluation_windows")
    subject_final = (
        report_windows.get("final") if isinstance(report_windows, dict) else None
    )
    baseline_final = (
        baseline_windows.get("final") if isinstance(baseline_windows, dict) else None
    )
    if not isinstance(subject_final, dict) or not isinstance(baseline_final, dict):
        return 0
    for key in ("window_ids", "example_ids"):
        subject_ids = subject_final.get(key)
        baseline_ids = baseline_final.get(key)
        if not (
            isinstance(subject_ids, list)
            and subject_ids
            and isinstance(baseline_ids, list)
            and len(subject_ids) == len(baseline_ids)
        ):
            continue
        subject_keys = [str(value) for value in subject_ids]
        baseline_keys = [str(value) for value in baseline_ids]
        if (
            len(set(subject_keys)) == len(subject_keys)
            and len(set(baseline_keys)) == len(baseline_keys)
            and subject_keys == baseline_keys
        ):
            return len(subject_keys)
    return 0


def _subject_slice_ids_are_disjoint(report: dict[str, Any]) -> bool:
    windows = report.get("evaluation_windows")
    preview = windows.get("preview") if isinstance(windows, dict) else None
    final = windows.get("final") if isinstance(windows, dict) else None
    if not isinstance(preview, dict) or not isinstance(final, dict):
        return False
    for key in ("window_ids", "example_ids"):
        preview_ids = preview.get(key)
        final_ids = final.get(key)
        if not (
            isinstance(preview_ids, list)
            and preview_ids
            and isinstance(final_ids, list)
            and final_ids
        ):
            continue
        preview_keys = [str(value) for value in preview_ids]
        final_keys = [str(value) for value in final_ids]
        return (
            len(set(preview_keys)) == len(preview_keys)
            and len(set(final_keys)) == len(final_keys)
            and set(preview_keys).isdisjoint(final_keys)
        )
    return False


def _coerce_bounds(bounds: Any) -> tuple[float, float] | None:
    if not isinstance(bounds, tuple | list) or len(bounds) != 2:
        return None
    lower, upper = bounds
    if not (
        isinstance(lower, int | float)
        and isinstance(upper, int | float)
        and math.isfinite(lower)
        and math.isfinite(upper)
    ):
        return None
    return float(lower), float(upper)


def _build_drift_ci(
    preview_ci: tuple[float, float] | None,
    final_ci: tuple[float, float] | None,
) -> tuple[float, float]:
    drift_ci = (float("nan"), float("nan"))
    preview_bounds = _coerce_bounds(preview_ci)
    final_bounds = _coerce_bounds(final_ci)
    if preview_bounds is not None and final_bounds is not None:
        lower_preview = max(preview_bounds[0], 1e-12)
        upper_preview = max(preview_bounds[1], 1e-12)
        drift_ci = (
            final_bounds[0] / upper_preview if upper_preview > 0 else float("nan"),
            final_bounds[1] / max(lower_preview, 1e-12),
        )
    return drift_ci


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _resolve_primary_metric_snapshot(
    report: dict[str, Any],
    baseline_ref: dict[str, Any],
) -> tuple[Any, Any, Any, float]:
    pm_blk = _report_metrics(report).get("primary_metric")
    if not isinstance(pm_blk, dict) or not pm_blk:
        try:
            pm_blk = compute_primary_metric_from_report(report)
        except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pm_blk = {}
    pm_prev = pm_blk.get("preview") if isinstance(pm_blk, dict) else float("nan")
    pm_fin = pm_blk.get("final") if isinstance(pm_blk, dict) else float("nan")
    pm_ratio = pm_blk.get("ratio_vs_baseline") if isinstance(pm_blk, dict) else None
    if not isinstance(pm_ratio, (int | float)):
        try:
            base_final = baseline_ref.get("primary_metric", {}).get("final")
            if (
                isinstance(pm_fin, (int | float))
                and isinstance(base_final, (int | float))
                and base_final > 0
            ):
                pm_ratio = float(pm_fin) / float(base_final)
        except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pm_ratio = float("nan")
    pm_preview_final_ratio = (
        float(pm_fin) / float(pm_prev)
        if isinstance(pm_fin, (int | float))
        and isinstance(pm_prev, (int | float))
        and pm_prev > 0
        else float("nan")
    )
    return pm_prev, pm_fin, pm_ratio, pm_preview_final_ratio


def _merge_metrics_stats_source(
    report: dict[str, Any],
    ppl_analysis: dict[str, Any],
) -> None:
    metrics_stats_source: dict[str, Any] = {}
    raw_metrics_stats = _report_metrics(report).get("stats", {}) or {}
    if isinstance(raw_metrics_stats, dict):
        metrics_stats_source = raw_metrics_stats
    stats_section = ppl_analysis.get("stats")
    if isinstance(metrics_stats_source, dict) and isinstance(stats_section, dict):
        for stats_key in (
            "requested_preview",
            "requested_final",
            "actual_preview",
            "actual_final",
            "coverage_ok",
        ):
            if stats_key in metrics_stats_source:
                stats_section[stats_key] = metrics_stats_source[stats_key]


def build_primary_metric_analysis(
    report: dict[str, Any],
    baseline_normalized: dict[str, Any],
    baseline_ref: dict[str, Any],
    dataset_info: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    (
        metrics,
        metrics_bootstrap,
        coverage_summary,
        window_plan_ctx,
        window_plan_profile,
    ) = _collect_bootstrap_context(report)
    (
        edited_preview,
        edited_final,
        ratio_vs_baseline,
        preview_final_ratio,
    ) = _resolve_primary_metric_snapshot(report, baseline_ref)
    preview_ci = None
    final_ci = None
    ratio_ci: tuple[float, float] | None
    ratio_ci, ratio_ci_source = _resolve_ratio_ci_from_run_metrics(metrics)

    paired_windows = 0
    unstable_ci_flag = _resolve_unstable_ci_flag(report, metrics, metrics_bootstrap)

    raw_logloss_delta = metrics.get("logloss_delta")
    logloss_delta = (
        float(raw_logloss_delta)
        if isinstance(raw_logloss_delta, int | float)
        else float("nan")
    )
    logloss_delta_ci: tuple[float, float] | None = _coerce_interval(
        metrics.get("logloss_delta_ci")
    )
    raw_slice_summary = metrics.get("preview_final_slice_delta_summary")
    if "paired_delta_summary" in metrics:
        raise ValueError(
            "metrics.paired_delta_summary is not supported; use "
            "metrics.preview_final_slice_delta_summary for independent preview/final "
            "slices."
        )
    if isinstance(raw_slice_summary, dict):
        preview_final_slice_delta_summary = dict(raw_slice_summary)
    else:
        preview_final_slice_delta_summary = {}

    (
        paired_windows,
        baseline_delta_mean,
        logloss_delta_ci,
        ratio_ci,
        ratio_ci_source,
    ) = _resolve_paired_window_analysis(
        report,
        baseline_normalized,
        metrics_bootstrap,
        logloss_delta_ci,
        ratio_ci,
        ratio_ci_source,
    )
    identical_id_pairs = _identical_final_id_pair_count(report, baseline_normalized)
    if paired_windows == 0:
        paired_windows = identical_id_pairs
    id_pairing_verified = identical_id_pairs > 0
    slice_ids_disjoint = _subject_slice_ids_are_disjoint(report)

    drift_ci = _build_drift_ci(preview_ci, final_ci)

    delta_mean = preview_final_slice_delta_summary.get("mean")
    degenerate_delta = preview_final_slice_delta_summary.get("degenerate", False)
    drift_ratio = preview_final_ratio

    delta_mean_float = (
        float(delta_mean)
        if isinstance(delta_mean, int | float) and math.isfinite(float(delta_mean))
        else None
    )
    if delta_mean_float is not None and not degenerate_delta:
        enforce_drift_ratio_identity(
            paired_windows,
            delta_mean_float,
            drift_ratio,
            window_plan_profile,
        )

    if _is_number(baseline_delta_mean) and _is_number(ratio_vs_baseline):
        expected_ratio_baseline = math.exp(float(baseline_delta_mean))
        tolerance = 5e-4 * max(1.0, abs(expected_ratio_baseline))
        if abs(expected_ratio_baseline - ratio_vs_baseline) > tolerance:
            raise ValueError(
                "Primary metric ratio mismatch: ratio_vs_baseline does not match "
                "the paired baseline log-loss delta."
            )

    if not (
        isinstance(ratio_vs_baseline, int | float) and math.isfinite(ratio_vs_baseline)
    ):
        try:
            if isinstance(baseline_delta_mean, int | float) and math.isfinite(
                baseline_delta_mean
            ):
                ratio_vs_baseline = math.exp(float(baseline_delta_mean))
                if not (
                    isinstance(ratio_ci, tuple | list) and len(ratio_ci) == 2
                ) and isinstance(edited_final, int | float):
                    ratio_ci = (float(edited_final), float(edited_final))
        except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pass

    enforce_ratio_ci_alignment(ratio_ci_source, ratio_ci, logloss_delta_ci)

    paired_windows_explicit = paired_windows > 0

    stats_payload: dict[str, Any] = {
        "metric_space": "log_nll",
        "bootstrap": metrics_bootstrap,
        "coverage": coverage_summary,
        "pairing": ratio_ci_source,
        "paired_windows": paired_windows,
        "window_pairing_reason": metrics.get("window_pairing_reason", None),
    }
    overlap_fraction = metrics.get("window_overlap_fraction")
    if (
        not isinstance(overlap_fraction, bool)
        and isinstance(overlap_fraction, int | float)
        and math.isfinite(float(overlap_fraction))
    ):
        stats_payload["window_overlap_fraction"] = float(overlap_fraction)
    elif id_pairing_verified and slice_ids_disjoint:
        stats_payload["window_overlap_fraction"] = 0.0
    match_fraction = metrics.get("window_match_fraction")
    if (
        not isinstance(match_fraction, bool)
        and isinstance(match_fraction, int | float)
        and math.isfinite(float(match_fraction))
    ):
        stats_payload["window_match_fraction"] = float(match_fraction)
    elif id_pairing_verified:
        stats_payload["window_match_fraction"] = 1.0
    if isinstance(raw_slice_summary, dict):
        stats_payload["preview_final_slice_delta_summary"] = (
            preview_final_slice_delta_summary
        )

    ppl_analysis = {
        "preview": edited_preview,
        "final": edited_final,
        "ratio_vs_baseline": ratio_vs_baseline
        if isinstance(ratio_vs_baseline, (int | float))
        else float("nan"),
        "preview_final_ratio": preview_final_ratio,
        "drift": preview_final_ratio,
        "preview_ci": None,
        "final_ci": None,
        "ratio_ci": ratio_ci,
        "degenerate": bool(
            isinstance(ratio_ci, list | tuple)
            and len(ratio_ci) == 2
            and all(isinstance(x, int | float) for x in ratio_ci)
            and abs(ratio_ci[0] - 1.0) < 1e-12
            and abs(ratio_ci[1] - 1.0) < 1e-12
        ),
        "unstable": bool(unstable_ci_flag),
        "drift_ci": drift_ci,
        "logloss_delta": logloss_delta,
        "logloss_delta_ci": logloss_delta_ci,
        "logloss_delta_paired_baseline": float(baseline_delta_mean)
        if _is_number(baseline_delta_mean)
        else None,
        "reduction": metrics.get("reduction"),
        "stats": stats_payload,
    }

    _merge_metrics_stats_source(report, ppl_analysis)
    paired_windows = _populate_stats_with_counts_and_coverage(
        report,
        dataset_info,
        coverage_summary,
        ppl_analysis,
        paired_windows,
        paired_windows_explicit,
    )

    auto_tier = _resolve_auto_tier(report)

    stats_payload = ppl_analysis.get("stats", {})
    enforce_pairing_and_coverage(
        stats_payload if isinstance(stats_payload, dict) else {},
        window_plan_profile,
        auto_tier,
    )

    if isinstance(window_plan_ctx, dict):
        ppl_analysis["window_plan"] = window_plan_ctx

    return ppl_analysis, window_plan_profile
