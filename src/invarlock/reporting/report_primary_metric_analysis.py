from __future__ import annotations

import copy
import math
from typing import Any

from invarlock.core import bootstrap as bootstrap_mod
from invarlock.core.auto_tuning import get_tier_policies
from invarlock.eval.primary_metric import compute_primary_metric_from_report

from .report_primary_metric_policy import (
    enforce_drift_ratio_identity,
    enforce_pairing_and_coverage,
    enforce_ratio_ci_alignment,
    fallback_paired_windows,
)
from .utils import _coerce_int, _coerce_interval, _pair_logloss_windows

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    RuntimeError,
    OSError,
)
_NUMERIC_EXCEPTIONS = (TypeError, ValueError, OverflowError)


def build_primary_metric_analysis(
    report: dict[str, Any],
    baseline_normalized: dict[str, Any],
    baseline_ref: dict[str, Any],
    dataset_info: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    edited_preview = float("nan")
    edited_final = float("nan")
    ratio_vs_baseline = float("nan")

    metrics_bootstrap_obj = (
        report["metrics"].get("bootstrap", {})
        if isinstance(report.get("metrics"), dict)
        else {}
    )
    metrics_bootstrap = (
        dict(metrics_bootstrap_obj) if isinstance(metrics_bootstrap_obj, dict) else {}
    )
    raw_coverage = metrics_bootstrap.get("coverage") if metrics_bootstrap else None
    coverage_summary = (
        copy.deepcopy(raw_coverage) if isinstance(raw_coverage, dict) else {}
    )
    window_plan_ctx = (
        report.get("metrics", {}).get("window_plan")
        if isinstance(report.get("metrics"), dict)
        else None
    )
    window_plan_profile = (
        str(window_plan_ctx.get("profile"))
        if isinstance(window_plan_ctx, dict) and window_plan_ctx.get("profile")
        else None
    )
    preview_ci = None
    final_ci = None
    ratio_ci = None
    ratio_ci_source = "run_metrics"

    if ratio_ci is None:
        try:
            dlci = _coerce_interval(report["metrics"].get("logloss_delta_ci"))
            if (
                isinstance(dlci, tuple | list)
                and len(dlci) == 2
                and all(isinstance(x, (int | float)) for x in dlci)
            ):
                lo, hi = float(dlci[0]), float(dlci[1])
                ratio_ci = (math.exp(lo), math.exp(hi))
                ratio_ci_source = "run_metrics"
        except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pass

    paired_windows = 0
    unstable_ci_flag = False
    try:
        rep_raw = metrics_bootstrap.get("replicates", metrics_bootstrap.get("n"))
        if rep_raw is not None and int(rep_raw) < 200:
            unstable_ci_flag = True
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        unstable_ci_flag = False

    try:
        tokens_prev = (
            report.get("metrics", {}).get("preview_total_tokens")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        tokens_fin = (
            report.get("metrics", {}).get("final_total_tokens")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        total_tokens = None
        if isinstance(tokens_prev, int | float) and isinstance(tokens_fin, int | float):
            total_tokens = int(tokens_prev) + int(tokens_fin)

        tier = "balanced"
        try:
            auto_cfg = (
                report.get("meta", {}).get("auto")
                if isinstance(report.get("meta"), dict)
                else None
            )
            if isinstance(auto_cfg, dict) and auto_cfg.get("tier"):
                tier = str(auto_cfg.get("tier")).lower()
        except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pass

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

    raw_logloss_delta = report["metrics"].get("logloss_delta")
    logloss_delta = (
        float(raw_logloss_delta)
        if isinstance(raw_logloss_delta, int | float)
        else float("nan")
    )
    logloss_delta_ci = _coerce_interval(report["metrics"].get("logloss_delta_ci"))
    raw_delta_summary = report["metrics"].get("paired_delta_summary", {})
    paired_delta_summary = (
        dict(raw_delta_summary) if isinstance(raw_delta_summary, dict) else {}
    )

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
    baseline_delta_mean = float("nan")
    if paired:
        paired_run, paired_base = paired
        paired_windows = len(paired_run)
        paired_weights: list[float] | None = None
        try:
            run_ids = (
                run_windows.get("window_ids") if isinstance(run_windows, dict) else None
            )
            run_w = (
                run_windows.get("token_counts")
                if isinstance(run_windows, dict)
                else None
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
                base_set = {
                    int(b_id) for b_id in base_ids if isinstance(b_id, int | float)
                }
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

        method = str(metrics_bootstrap.get("method", "percentile")).lower()
        replicates = int(
            metrics_bootstrap.get(
                "replicates", metrics_bootstrap.get("n", 1000) or 1000
            )
        )
        alpha = float(metrics_bootstrap.get("alpha", 0.05) or 0.05)
        seed = int(metrics_bootstrap.get("seed", 0) or 0)
        ci_method = "percentile"
        if "bca" in method:
            ci_method = "bca"
        if replicates > 0:
            try:
                delta_ci = bootstrap_mod.compute_paired_delta_log_ci(
                    paired_run,
                    paired_base,
                    weights=paired_weights,
                    method=ci_method,
                    replicates=replicates,
                    alpha=alpha,
                    seed=seed + 503,
                )
                if isinstance(delta_ci, tuple | list) and len(delta_ci) == 2:
                    delta_ci = (float(delta_ci[0]), float(delta_ci[1]))
                logloss_delta_ci = delta_ci
                ratio_ci = bootstrap_mod.logspace_to_ratio_ci(delta_ci)
                ratio_ci_source = "paired_baseline"
                try:
                    run_ids = (
                        run_windows.get("window_ids")
                        if isinstance(run_windows, dict)
                        else None
                    )
                    run_ll = (
                        run_windows.get("logloss")
                        if isinstance(run_windows, dict)
                        else None
                    )
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
                        run_windows.get("token_counts")
                        if isinstance(run_windows, dict)
                        else None
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
                            if isinstance(b_id, int | float) and isinstance(
                                b_val, int | float
                            ):
                                base_map[int(b_id)] = float(b_val)
                        sum_w = 0.0
                        sum_dw = 0.0
                        for r_id, r_val, w in zip(run_ids, run_ll, run_w, strict=False):
                            if not (
                                isinstance(r_id, int | float)
                                and isinstance(r_val, int | float)
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
            except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
                ratio_ci_source = "run_metrics"

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

    def _is_number(value: Any) -> bool:
        return isinstance(value, int | float) and math.isfinite(float(value))

    delta_mean = paired_delta_summary.get("mean")
    degenerate_delta = paired_delta_summary.get("degenerate", False)
    drift_ratio = (
        edited_final / edited_preview
        if _is_number(edited_final)
        and _is_number(edited_preview)
        and edited_preview > 0
        else float("nan")
    )

    ratio_from_delta = None
    delta_mean_float = (
        float(delta_mean)
        if isinstance(delta_mean, int | float) and math.isfinite(float(delta_mean))
        else None
    )
    if delta_mean_float is not None and not degenerate_delta:
        ratio_from_delta = enforce_drift_ratio_identity(
            paired_windows,
            delta_mean_float,
            drift_ratio,
            window_plan_profile,
        )

    if (
        ratio_from_delta is not None
        and _is_number(baseline_delta_mean)
        and _is_number(ratio_vs_baseline)
    ):
        expected_ratio_baseline = math.exp(float(baseline_delta_mean))
        tolerance = 5e-4 * max(1.0, abs(expected_ratio_baseline))
        if abs(expected_ratio_baseline - ratio_vs_baseline) > tolerance:
            pass

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

    paired_windows = fallback_paired_windows(paired_windows, coverage_summary)
    try:
        paired_windows_signal = (
            report.get("metrics", {}).get("paired_windows")
            if isinstance(report.get("metrics"), dict)
            else None
        )
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        paired_windows_signal = None
    paired_windows_signal_int = _coerce_int(paired_windows_signal)
    if paired_windows_signal_int is not None and paired_windows_signal_int >= 0:
        paired_windows = paired_windows_signal_int

    try:
        pm_blk = (
            report.get("metrics", {}).get("primary_metric")
            if isinstance(report.get("metrics"), dict)
            else None
        )
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pm_blk = None
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
    ppl_analysis = {
        "preview": pm_prev,
        "final": pm_fin,
        "ratio_vs_baseline": pm_ratio
        if isinstance(pm_ratio, (int | float))
        else float("nan"),
        "preview_final_ratio": pm_preview_final_ratio,
        "drift": pm_preview_final_ratio,
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
        "reduction": report["metrics"].get("reduction")
        if isinstance(report.get("metrics"), dict)
        else None,
        "stats": {
            "metric_space": "log_nll",
            "bootstrap": metrics_bootstrap,
            "coverage": coverage_summary,
            "pairing": ratio_ci_source,
            "paired_windows": paired_windows,
            "window_overlap_fraction": report["metrics"].get(
                "window_overlap_fraction", float("nan")
            ),
            "window_match_fraction": report["metrics"].get(
                "window_match_fraction", float("nan")
            ),
            "window_pairing_reason": report["metrics"].get(
                "window_pairing_reason", None
            ),
            "paired_delta_summary": paired_delta_summary,
        },
    }

    metrics_stats_source: dict[str, Any] = {}
    if isinstance(report.get("metrics"), dict):
        raw_metrics_stats = report["metrics"].get("stats", {}) or {}
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

    try:
        stats_obj = ppl_analysis.get("stats", {})
        if isinstance(stats_obj, dict):

            def _as_count(value: Any) -> int | None:
                if value is None or isinstance(value, bool):
                    return None
                if isinstance(value, int):
                    return int(value) if value >= 0 else None
                if isinstance(value, float) and math.isfinite(value):
                    if abs(value - round(value)) > 1e-9 or value < 0:
                        return None
                    return int(round(value))
                return None

            data_cfg = report.get("data", {}) if isinstance(report, dict) else {}
            data_cfg = data_cfg if isinstance(data_cfg, dict) else {}
            windows_cfg = (
                dataset_info.get("windows", {})
                if isinstance(dataset_info, dict)
                else {}
            )
            windows_cfg = windows_cfg if isinstance(windows_cfg, dict) else {}

            req_prev = _as_count(stats_obj.get("requested_preview"))
            if req_prev is None:
                req_prev = _as_count(data_cfg.get("preview_n"))
            if req_prev is None:
                req_prev = _as_count(windows_cfg.get("preview"))

            req_fin = _as_count(stats_obj.get("requested_final"))
            if req_fin is None:
                req_fin = _as_count(data_cfg.get("final_n"))
            if req_fin is None:
                req_fin = _as_count(windows_cfg.get("final"))

            eval_windows = (
                report.get("evaluation_windows", {}) if isinstance(report, dict) else {}
            )
            eval_windows = eval_windows if isinstance(eval_windows, dict) else {}

            def _len_ids(section: Any) -> int | None:
                if not isinstance(section, dict):
                    return None
                ids = section.get("window_ids")
                if isinstance(ids, list):
                    return int(len(ids))
                return None

            act_prev = _as_count(stats_obj.get("actual_preview"))
            if act_prev is None:
                act_prev = _len_ids(eval_windows.get("preview"))
            if act_prev is None:
                cov_prev = (
                    coverage_summary.get("preview")
                    if isinstance(coverage_summary, dict)
                    else None
                )
                if isinstance(cov_prev, dict):
                    act_prev = _as_count(cov_prev.get("used"))
            if act_prev is None:
                act_prev = req_prev

            act_fin = _as_count(stats_obj.get("actual_final"))
            if act_fin is None:
                act_fin = _len_ids(eval_windows.get("final"))
            if act_fin is None:
                cov_fin = (
                    coverage_summary.get("final")
                    if isinstance(coverage_summary, dict)
                    else None
                )
                if isinstance(cov_fin, dict):
                    act_fin = _as_count(cov_fin.get("used"))
                elif isinstance(coverage_summary, dict):
                    act_fin = _as_count(coverage_summary.get("used"))
            if act_fin is None:
                act_fin = req_fin

            if req_prev is not None:
                stats_obj["requested_preview"] = req_prev
            if req_fin is not None:
                stats_obj["requested_final"] = req_fin
            if act_prev is not None:
                stats_obj["actual_preview"] = act_prev
            if act_fin is not None:
                stats_obj["actual_final"] = act_fin

            if "coverage_ok" not in stats_obj:
                if (
                    isinstance(req_prev, int)
                    and isinstance(req_fin, int)
                    and isinstance(act_prev, int)
                    and isinstance(act_fin, int)
                ):
                    stats_obj["coverage_ok"] = (act_prev >= req_prev) and (
                        act_fin >= req_fin
                    )
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    auto_tier = "balanced"
    try:
        auto_cfg = (
            report.get("meta", {}).get("auto")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(auto_cfg, dict) and auto_cfg.get("tier"):
            auto_tier = str(auto_cfg.get("tier")).lower()
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        auto_tier = "balanced"

    stats_payload = ppl_analysis.get("stats", {})
    enforce_pairing_and_coverage(
        stats_payload if isinstance(stats_payload, dict) else {},
        window_plan_profile,
        auto_tier,
    )

    if isinstance(window_plan_ctx, dict):
        ppl_analysis["window_plan"] = window_plan_ctx

    return ppl_analysis, window_plan_profile
