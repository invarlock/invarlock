# mypy: ignore-errors
# ruff: noqa: F821
"""Implementation body for reporting.report_builder.make_report."""

from __future__ import annotations

from typing import Any

from invarlock.reporting.report_types import RunReport


def make_report_impl(
    report: RunReport,
    baseline: RunReport | dict[str, Any],
) -> dict[str, Any]:
    """Delegated implementation of report_builder.make_report."""
    from invarlock.reporting import report_builder as builder_mod

    # Keep runtime lookup dynamic so report_builder monkeypatches in tests continue to apply.
    globals().update(builder_mod.__dict__)

    NON_FATAL_EXCEPTIONS = (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
    )
    NUMERIC_EXCEPTIONS = (TypeError, ValueError, OverflowError)

    report = _normalize_and_validate_report(report)

    # Normalize baseline input
    baseline_raw = baseline
    baseline_normalized = _normalize_baseline(baseline_raw)
    baseline_report: RunReport | None = None
    try:
        if (
            isinstance(baseline_raw, dict)
            and "meta" in baseline_raw
            and "metrics" in baseline_raw
            and "edit" in baseline_raw
        ):
            baseline_report = _normalize_and_validate_report(baseline_raw)
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover - baseline compare is best-effort
        baseline_report = None

    # Extract core metadata with full seed bundle
    meta = _extract_report_meta(report)

    # Propagate environment flags captured in the RunReport (e.g., deterministic algos,
    # TF32 controls, MPS/CUDA availability). This is useful for auditability and
    # reproducibility of evaluation runs.
    try:
        env_flags = (
            report.get("meta", {}).get("env_flags")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(env_flags, dict) and env_flags:
            meta["env_flags"] = env_flags
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    # Determinism preset (CI/Release provenance) when present.
    try:
        det = (
            report.get("meta", {}).get("determinism")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(det, dict) and det:
            meta["determinism"] = det
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    # Execution profile provenance when available via run context.
    try:
        ctx = report.get("context") if isinstance(report, dict) else None
        ctx_profile = (
            str(ctx.get("profile") or "").strip().lower()
            if isinstance(ctx, dict)
            else ""
        )
        if ctx_profile:
            meta["profile"] = ctx_profile
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    tokenizer_hash_meta = report["meta"].get("tokenizer_hash")
    if not tokenizer_hash_meta:
        dataset_section = report.get("data", {})
        if isinstance(dataset_section, dict):
            tokenizer_hash_meta = dataset_section.get("tokenizer_hash")
    if isinstance(tokenizer_hash_meta, str) and tokenizer_hash_meta:
        meta["tokenizer_hash"] = tokenizer_hash_meta

    model_profile_meta = report["meta"].get("model_profile")
    if isinstance(model_profile_meta, dict) and model_profile_meta:
        meta["model_profile"] = model_profile_meta

    cuda_flags = report["meta"].get("cuda_flags")
    if isinstance(cuda_flags, dict) and cuda_flags:
        meta["cuda_flags"] = cuda_flags

    # Extract auto-tuning configuration
    auto_config = report["meta"].get("auto")
    if auto_config:
        auto = {
            "tier": auto_config.get("tier", "balanced"),
            "probes_used": auto_config.get("probes", auto_config.get("probes_used", 0)),
            "target_pm_ratio": auto_config.get("target_pm_ratio"),
        }
    else:
        auto = {"tier": "none", "probes_used": 0, "target_pm_ratio": None}

    # Extract dataset configuration and compute hashes
    dataset_info = _extract_dataset_info(report)
    try:
        if isinstance(dataset_info, dict):
            windows = dataset_info.get("windows")
            if isinstance(windows, dict):
                windows.setdefault("stats", {})
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    # Baseline reference (PM-only). Derive a primary_metric snapshot from baseline windows.
    # Prefer explicit baseline primary_metric when provided; otherwise compute from windows
    baseline_pm = None
    try:
        bm = (
            baseline_raw.get("metrics", {}).get("primary_metric")
            if isinstance(baseline_raw.get("metrics"), dict)
            else None
        )
        if isinstance(bm, dict) and bm:
            baseline_pm = bm
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        baseline_pm = None
    if not isinstance(baseline_pm, dict) or not baseline_pm:
        try:
            baseline_pm = compute_primary_metric_from_report(baseline_normalized)
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            baseline_pm = {"kind": "ppl_causal", "final": float("nan")}
    baseline_ref = {
        "run_id": baseline_normalized.get("run_id", "unknown"),
        "model_id": baseline_normalized.get("model_id", report["meta"]["model_id"]),
        "primary_metric": {
            "kind": baseline_pm.get("kind", "ppl_causal"),
            "final": baseline_pm.get("final", float("nan")),
        },
    }
    # Propagate baseline tokenizer hash for verify-time linting when available
    baseline_tok_hash = baseline_normalized.get("tokenizer_hash")
    if isinstance(baseline_tok_hash, str) and baseline_tok_hash:
        baseline_ref["tokenizer_hash"] = baseline_tok_hash

    # Primary-metric analysis (PM-only)
    ppl_metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
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
    # PM-only fallback: derive ratio_ci from logloss_delta_ci when available
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
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pass
    paired_windows = 0
    # UX hint: mark CI as unstable for very low replicate counts or insufficient tokens
    unstable_ci_flag = False
    try:
        rep_raw = metrics_bootstrap.get("replicates", metrics_bootstrap.get("n"))
        if rep_raw is not None and int(rep_raw) < 200:
            unstable_ci_flag = True
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        unstable_ci_flag = False
    # Also consider token-count floor from tier policy when available
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
        # Resolve tier
        tier = "balanced"
        try:
            auto_cfg = (
                report.get("meta", {}).get("auto")
                if isinstance(report.get("meta"), dict)
                else None
            )
            if isinstance(auto_cfg, dict) and auto_cfg.get("tier"):
                tier = str(auto_cfg.get("tier")).lower()
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
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
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
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
        if isinstance(baseline_normalized.get("evaluation_windows"), dict)
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
                    except NUMERIC_EXCEPTIONS:
                        continue
                    if not math.isfinite(wv):
                        continue
                    weights.append(float(max(wv, 0.0)))
                if weights:
                    paired_weights = weights
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            paired_weights = None
        method = str(metrics_bootstrap.get("method", "percentile")).lower()
        replicates = int(
            metrics_bootstrap.get(
                "replicates", metrics_bootstrap.get("n", 1000) or 1000
            )
        )
        alpha = float(metrics_bootstrap.get("alpha", 0.05) or 0.05)
        seed = int(metrics_bootstrap.get("seed", 0) or 0)
        # Default to percentile for deterministic behavior; enable BCa only when requested
        ci_method = "percentile"
        try:
            if "bca" in method:
                ci_method = "bca"
            else:
                # Opt-in via env flag and sufficiently large sample
                use_bca_flag = str(
                    os.environ.get("INVARLOCK_BOOTSTRAP_BCA", "")
                ).strip().lower() in {"1", "true", "yes", "on"}
                if use_bca_flag and paired_windows >= 200:
                    ci_method = "bca"
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pass
        if replicates > 0:
            try:
                delta_ci = compute_paired_delta_log_ci(
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
                ratio_ci = logspace_to_ratio_ci(delta_ci)
                ratio_ci_source = "paired_baseline"
                # Compute token-weighted paired mean ΔlogNLL vs baseline for identity checks
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
                            except NON_FATAL_EXCEPTIONS:  # pragma: no cover
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
                except NON_FATAL_EXCEPTIONS:  # pragma: no cover
                    baseline_delta_mean = float("nan")
            except NON_FATAL_EXCEPTIONS:  # pragma: no cover
                ratio_ci_source = "run_metrics"

    def _finite_bounds(bounds: tuple[float, float]) -> bool:
        return (
            isinstance(bounds, tuple | list)
            and len(bounds) == 2
            and all(isinstance(v, int | float) and math.isfinite(v) for v in bounds)
        )

    drift_ci = (float("nan"), float("nan"))
    if _finite_bounds(preview_ci) and _finite_bounds(final_ci):
        lower_preview = max(preview_ci[0], 1e-12)
        upper_preview = max(preview_ci[1], 1e-12)
        drift_ci = (
            final_ci[0] / upper_preview if upper_preview > 0 else float("nan"),
            final_ci[1] / max(lower_preview, 1e-12),
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
    if _is_number(delta_mean) and not degenerate_delta:
        ratio_from_delta = _enforce_drift_ratio_identity(
            paired_windows, float(delta_mean), drift_ratio, window_plan_profile
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

    # Fallback: if we could not compute a finite ratio, but we did compute a paired
    # baseline delta, use exp(delta) as an identity-consistent ratio. This covers
    # tiny runs where ppl_* fields are absent and PM-only windows are identical.
    if not (
        isinstance(ratio_vs_baseline, int | float) and math.isfinite(ratio_vs_baseline)
    ):
        try:
            if isinstance(baseline_delta_mean, int | float) and math.isfinite(
                baseline_delta_mean
            ):
                ratio_vs_baseline = math.exp(float(baseline_delta_mean))
                # Provide a degenerate CI if none was computed
                if not (
                    isinstance(ratio_ci, tuple | list) and len(ratio_ci) == 2
                ) and isinstance(edited_final, int | float):
                    ratio_ci = (float(edited_final), float(edited_final))
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pass

    _enforce_ratio_ci_alignment(ratio_ci_source, ratio_ci, logloss_delta_ci)

    paired_windows = _fallback_paired_windows(paired_windows, coverage_summary)
    # Prefer runner-reported paired window count when available (signal used for
    # CI/Release enforcement); fall back to evidence-based pairing or coverage
    # heuristics when the metric is missing.
    try:
        paired_windows_signal = (
            report.get("metrics", {}).get("paired_windows")
            if isinstance(report.get("metrics"), dict)
            else None
        )
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        paired_windows_signal = None
    paired_windows_signal_int = _coerce_int(paired_windows_signal)
    if paired_windows_signal_int is not None and paired_windows_signal_int >= 0:
        paired_windows = paired_windows_signal_int

    # Primary-metric stats for gating/summary (PM-only)
    try:
        pm_blk = (
            report.get("metrics", {}).get("primary_metric")
            if isinstance(report.get("metrics"), dict)
            else None
        )
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pm_blk = None
    if not isinstance(pm_blk, dict) or not pm_blk:
        try:
            pm_blk = compute_primary_metric_from_report(report)
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
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
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
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

    metrics_stats_source = {}
    if isinstance(report.get("metrics"), dict):
        metrics_stats_source = report["metrics"].get("stats", {}) or {}
    if isinstance(metrics_stats_source, dict):
        for key in (
            "requested_preview",
            "requested_final",
            "actual_preview",
            "actual_final",
            "coverage_ok",
        ):
            if key in metrics_stats_source:
                ppl_analysis["stats"][key] = metrics_stats_source[key]

    # Derive requested/actual window counts for auditability when runners do not
    # emit a metrics.stats block (normalization may also drop it).
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
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    _enforce_pairing_and_coverage(
        ppl_analysis.get("stats", {}),
        window_plan_profile,
        auto.get("tier", "balanced"),
    )

    if isinstance(window_plan_ctx, dict):
        ppl_analysis["window_plan"] = window_plan_ctx

    # Extract invariant status
    invariants = _extract_invariants(report, baseline=baseline_report)

    # Extract spectral analysis
    spectral = _extract_spectral_analysis(report, baseline_normalized)

    # Extract RMT analysis
    rmt = _extract_rmt_analysis(report, baseline_normalized)

    # Extract variance guard info
    variance = _extract_variance_analysis(report)

    # Extract structural deltas
    structure = _extract_structural_deltas(report)
    compression_diag = structure.get("compression_diagnostics", {})
    structure["compression_diagnostics"] = compression_diag

    # Extract effective policies used
    policies = _extract_effective_policies(report)
    variance_policy = policies.get("variance")
    guard_variance_policy = None
    for guard in report.get("guards", []):
        if guard.get("name", "").lower() == "variance" and isinstance(
            guard.get("policy"), dict
        ):
            guard_variance_policy = guard.get("policy")
            break

    variance_policy_digest = ""
    if isinstance(variance_policy, dict):
        variance_policy_digest = _compute_variance_policy_digest(variance_policy)
        if not variance_policy_digest and isinstance(guard_variance_policy, dict):
            variance_policy_digest = _compute_variance_policy_digest(
                guard_variance_policy
            )
            if variance_policy_digest:
                for key in VARIANCE_CANONICAL_KEYS:
                    if (
                        isinstance(guard_variance_policy, dict)
                        and key in guard_variance_policy
                        and key not in variance_policy
                    ):
                        variance_policy[key] = guard_variance_policy[key]
        if variance_policy_digest:
            policies["variance"]["policy_digest"] = variance_policy_digest

    # Resolve tier/profile policy (canonical) and merge observed guard policies.
    profile = None
    explicit_overrides = None
    try:
        ctx = report.get("context") if isinstance(report, dict) else None
        if isinstance(ctx, dict) and ctx.get("profile"):
            profile = str(ctx.get("profile"))
    except NON_FATAL_EXCEPTIONS:
        profile = None
    try:
        window_plan = (
            report.get("metrics", {}).get("window_plan")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        if (
            profile is None
            and isinstance(window_plan, dict)
            and window_plan.get("profile")
        ):
            profile = str(window_plan.get("profile"))
    except NON_FATAL_EXCEPTIONS:
        profile = None
    try:
        meta_cfg = (
            report.get("meta", {}).get("config")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(meta_cfg, dict) and isinstance(meta_cfg.get("guards"), dict):
            explicit_overrides = meta_cfg.get("guards")
        if explicit_overrides is None and isinstance(report.get("config"), dict):
            cfg2 = report.get("config")
            if isinstance(cfg2.get("guards"), dict):
                explicit_overrides = cfg2.get("guards")
    except NON_FATAL_EXCEPTIONS:
        explicit_overrides = None

    resolved_policy = _build_resolved_policies(
        auto.get("tier", "balanced"),
        spectral,
        rmt,
        variance,
        profile=profile,
        explicit_overrides=explicit_overrides,
    )
    overrides_list = _extract_policy_overrides(report)
    resolved_digest = _compute_policy_digest(
        {
            "resolved_policy": resolved_policy,
            "overrides": overrides_list,
        }
    )
    policy_provenance = {
        "tier": auto.get("tier", "balanced"),
        "overrides": overrides_list,
        "policy_digest": resolved_digest,
    }
    auto["policy_digest"] = resolved_digest

    for guard_name in ("spectral", "rmt", "variance"):
        if guard_name in resolved_policy:
            policies[guard_name] = copy.deepcopy(resolved_policy[guard_name])
            if guard_name == "variance" and variance_policy_digest:
                policies[guard_name]["policy_digest"] = variance_policy_digest

    plugin_provenance = report.get("meta", {}).get("plugins", {})
    edit_metadata = _extract_edit_metadata(report, plugin_provenance)

    # Extract telemetry (latency, memory, etc.)
    telemetry: dict[str, Any] = {}
    metrics_section = report.get("metrics", {})
    if isinstance(metrics_section, dict):
        for key in (
            "latency_ms_per_tok",
            "memory_mb_peak",
            "gpu_memory_mb_peak",
            "gpu_memory_reserved_mb_peak",
            "throughput_tok_per_s",
        ):
            value = metrics_section.get(key)
            if isinstance(value, int | float) and math.isfinite(value):
                telemetry[key] = float(value)

        for key in ("preview_total_tokens", "final_total_tokens"):
            value = metrics_section.get(key)
            if isinstance(value, int | float) and value >= 0:
                telemetry[key] = float(value)
        for key in (
            "masked_tokens_total",
            "masked_tokens_preview",
            "masked_tokens_final",
        ):
            value = metrics_section.get(key)
            if isinstance(value, int | float) and value >= 0:
                telemetry[key] = float(value)

        edge_ctx = metrics_section.get("edge_device")
        if isinstance(edge_ctx, dict):
            telemetry["edge_device"] = edge_ctx

    device_name = meta.get("device")
    if device_name:
        telemetry.setdefault("device", device_name)

    # Build the evaluation report
    window_capacity_ctx = (
        report.get("metrics", {}).get("window_capacity")
        if isinstance(report.get("metrics"), dict)
        else None
    )
    window_plan_ctx = (
        report.get("metrics", {}).get("window_plan")
        if isinstance(report.get("metrics"), dict)
        else None
    )

    report_artifacts = (
        report.get("artifacts", {}) if isinstance(report.get("artifacts"), dict) else {}
    )
    artifacts_payload = {
        "events_path": report_artifacts.get("events_path", ""),
        "report_path": report_artifacts.get(
            "report_path", report_artifacts.get("logs_path", "")
        ),
        "generated_at": datetime.now().isoformat(),
    }
    masks_path = report_artifacts.get("masks_path")
    if isinstance(masks_path, str) and masks_path:
        artifacts_payload["masks_path"] = masks_path

    raw_guard_ctx = report.get("guard_overhead")
    guard_overhead_section, _ = _prepare_guard_overhead_section(raw_guard_ctx)

    # Add schedule digest to provenance/overhead for auditability of schedule reuse
    try:
        final_windows_ctx = (
            report.get("evaluation_windows", {}).get("final", {})
            if isinstance(report.get("evaluation_windows"), dict)
            else {}
        )
        window_ids = final_windows_ctx.get("window_ids")
        if isinstance(window_ids, list) and window_ids:
            import hashlib as _hashlib

            h = _hashlib.blake2s(digest_size=16)
            for wid in window_ids:
                try:
                    h.update(int(wid).to_bytes(8, "little", signed=True))
                except NON_FATAL_EXCEPTIONS:  # pragma: no cover
                    h.update(str(wid).encode("utf-8", "ignore"))
            schedule_digest = h.hexdigest()
            guard_overhead_section["schedule_digest"] = schedule_digest
        else:
            schedule_digest = None
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        schedule_digest = None

    policy_provenance["resolved_at"] = artifacts_payload["generated_at"]

    current_run_id = _generate_run_id(report)
    provenance = _build_provenance_block(
        report,
        baseline_raw,
        baseline_ref,
        artifacts_payload,
        policy_provenance,
        schedule_digest,
        ppl_analysis,
        current_run_id,
    )

    # Prepare MoE section (observability; non-gating)
    moe_section: dict[str, Any] = {}
    try:
        run_moe = (
            report.get("metrics", {}).get("moe")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        base_moe = None
        # Try raw baseline first (dict with optional 'moe')
        if isinstance(baseline_raw, dict):
            try:
                base_moe = baseline_raw.get("moe")
            except NON_FATAL_EXCEPTIONS:  # pragma: no cover
                base_moe = None
        # Then normalized baseline variants
        if (not isinstance(base_moe, dict) or not base_moe) and isinstance(
            baseline_normalized, dict
        ):
            try:
                bm = baseline_normalized.get("moe")
                if isinstance(bm, dict) and bm:
                    base_moe = bm
                else:
                    mx = (
                        baseline_normalized.get("metrics")
                        if isinstance(baseline_normalized.get("metrics"), dict)
                        else None
                    )
                    if isinstance(mx, dict):
                        base_moe = mx.get("moe")
            except NON_FATAL_EXCEPTIONS:  # pragma: no cover
                pass
        if isinstance(run_moe, dict) and run_moe:
            # Copy selected fields
            for key in (
                "top_k",
                "capacity_factor",
                "expert_drop_rate",
                "load_balance_loss",
                "router_entropy",
            ):
                val = run_moe.get(key)
                if isinstance(val, int | float):
                    moe_section[key] = float(val)
            # Utilization summary
            util = run_moe.get("utilization")
            if isinstance(util, list) and util:
                try:
                    util_vals = [float(x) for x in util]
                    moe_section["utilization_mean"] = float(
                        sum(util_vals) / max(1, len(util_vals))
                    )
                    moe_section["utilization_count"] = int(len(util_vals))
                except NON_FATAL_EXCEPTIONS:  # pragma: no cover
                    pass
            # Deltas vs baseline (if available)
            if isinstance(base_moe, dict) and base_moe:
                for key in ("load_balance_loss", "router_entropy"):
                    rv = run_moe.get(key)
                    bv = base_moe.get(key)
                    if isinstance(rv, int | float) and isinstance(bv, int | float):
                        moe_section[f"delta_{key}"] = float(rv) - float(bv)
                bu = base_moe.get("utilization")
                if isinstance(util, list) and isinstance(bu, list) and util and bu:
                    try:
                        util_vals = [float(x) for x in util]
                        bu_vals = [float(x) for x in bu]
                        mu = float(sum(util_vals) / len(util_vals))
                        mb = float(sum(bu_vals) / len(bu_vals))
                        moe_section["delta_utilization_mean"] = mu - mb
                    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
                        pass
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        moe_section = {}

    # Build dataset capacity context for gating floors
    capacity_tokens: int | None = None
    capacity_examples: int | None = None
    try:
        if isinstance(window_capacity_ctx, dict):
            tv = window_capacity_ctx.get("total_tokens")
            if isinstance(tv, int | float):
                capacity_tokens = int(tv)
            ex = (
                window_capacity_ctx.get("available_unique")
                or window_capacity_ctx.get("available_nonoverlap")
                or window_capacity_ctx.get("candidate_limit")
            )
            if isinstance(ex, int | float):
                capacity_examples = int(ex)
        # Fallback: sum of configured windows
        if capacity_examples is None:
            try:
                capacity_examples = int(
                    dataset_info.get("windows", {}).get("preview", 0)
                ) + int(dataset_info.get("windows", {}).get("final", 0))
            except NON_FATAL_EXCEPTIONS:  # pragma: no cover
                capacity_examples = None
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        capacity_tokens = None
        capacity_examples = None

    pm_acceptance_range = _resolve_pm_acceptance_range_from_report(report)
    pm_drift_band = _resolve_pm_drift_band_from_report(report)
    tiny_relax = _resolve_tiny_relax_from_report(report)

    # Primary metric tail evidence and gate evaluation (ΔlogNLL vs baseline, per-window).
    pm_tail_result: dict[str, Any] = {}
    try:
        pm_kind = None
        try:
            pm_block = (
                report.get("metrics", {}).get("primary_metric")
                if isinstance(report.get("metrics"), dict)
                else None
            )
            if isinstance(pm_block, dict):
                pm_kind = pm_block.get("kind")
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pm_kind = None

        pm_tail_policy: dict[str, Any] = {}
        try:
            metrics_pol = (
                resolved_policy.get("metrics", {})
                if isinstance(resolved_policy, dict)
                else {}
            )
            if isinstance(metrics_pol, dict) and isinstance(
                metrics_pol.get("pm_tail"), dict
            ):
                pm_tail_policy = dict(metrics_pol.get("pm_tail") or {})
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pm_tail_policy = {}

        deltas: list[float] = []
        weights: list[float] = []
        if _is_ppl_kind(pm_kind):
            run_windows = (
                report.get("evaluation_windows", {}).get("final", {})
                if isinstance(report.get("evaluation_windows"), dict)
                else {}
            )
            base_windows = (
                baseline_normalized.get("evaluation_windows", {}).get("final", {})
                if isinstance(baseline_normalized.get("evaluation_windows"), dict)
                else {}
            )
            run_ids = (
                run_windows.get("window_ids") if isinstance(run_windows, dict) else None
            )
            run_ll = (
                run_windows.get("logloss") if isinstance(run_windows, dict) else None
            )
            run_tc = (
                run_windows.get("token_counts")
                if isinstance(run_windows, dict)
                else None
            )
            base_ids = (
                base_windows.get("window_ids")
                if isinstance(base_windows, dict)
                else None
            )
            base_ll = (
                base_windows.get("logloss") if isinstance(base_windows, dict) else None
            )
            if (
                isinstance(run_ids, list)
                and isinstance(run_ll, list)
                and isinstance(base_ids, list)
                and isinstance(base_ll, list)
            ):
                base_map: dict[int, float] = {}
                for b_id, b_val in zip(base_ids, base_ll, strict=False):
                    if isinstance(b_id, int | float) and isinstance(b_val, int | float):
                        base_map[int(b_id)] = float(b_val)
                for idx, (r_id, r_val) in enumerate(zip(run_ids, run_ll, strict=False)):
                    if not (
                        isinstance(r_id, int | float) and isinstance(r_val, int | float)
                    ):
                        continue
                    key = int(r_id)
                    if key not in base_map:
                        continue
                    dv = float(r_val) - base_map[key]
                    if math.isfinite(dv):
                        deltas.append(float(dv))
                        if isinstance(run_tc, list) and idx < len(run_tc):
                            try:
                                wv = float(run_tc[idx])
                            except NUMERIC_EXCEPTIONS:
                                wv = 0.0
                            weights.append(float(max(wv, 0.0)))

        pm_tail_result = evaluate_metric_tail(
            deltas=deltas,
            weights=weights if (weights and len(weights) == len(deltas)) else None,
            policy=pm_tail_policy,
        )
        pm_tail_result["source"] = "paired_baseline.final"
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pm_tail_result = {"mode": "warn", "evaluated": False, "passed": True}

    validation_kwargs = {
        "ppl": ppl_analysis,
        "spectral": spectral,
        "rmt": rmt,
        "invariants": invariants,
        "tier": auto.get("tier", "balanced"),
        "_ppl_metrics": ppl_metrics,
        "target_ratio": auto.get("target_pm_ratio"),
        "guard_overhead": guard_overhead_section,
        "primary_metric": report.get("metrics", {}).get("primary_metric")
        if isinstance(report.get("metrics"), dict)
        else None,
        "moe": moe_section,
        "dataset_capacity": {
            "tokens_available": capacity_tokens,
            "examples_available": capacity_examples,
        },
    }
    try:
        if (
            "pm_acceptance_range"
            in inspect.signature(_compute_validation_flags).parameters
        ):
            validation_kwargs["pm_acceptance_range"] = pm_acceptance_range
    except (
        NON_FATAL_EXCEPTIONS
    ):  # pragma: no cover - defensive against patched functions
        validation_kwargs["pm_acceptance_range"] = pm_acceptance_range

    try:
        if "pm_drift_band" in inspect.signature(_compute_validation_flags).parameters:
            validation_kwargs["pm_drift_band"] = pm_drift_band
    except (
        NON_FATAL_EXCEPTIONS
    ):  # pragma: no cover - defensive against patched functions
        validation_kwargs["pm_drift_band"] = pm_drift_band

    try:
        if "pm_tail" in inspect.signature(_compute_validation_flags).parameters:
            validation_kwargs["pm_tail"] = pm_tail_result
    except (
        NON_FATAL_EXCEPTIONS
    ):  # pragma: no cover - defensive against patched functions
        validation_kwargs["pm_tail"] = pm_tail_result

    try:
        if "tiny_relax" in inspect.signature(_compute_validation_flags).parameters:
            validation_kwargs["tiny_relax"] = tiny_relax
    except (
        NON_FATAL_EXCEPTIONS
    ):  # pragma: no cover - defensive against patched functions
        validation_kwargs["tiny_relax"] = tiny_relax

    validation_flags = _compute_validation_flags(**validation_kwargs)

    # Enforce validation key allow-list to prevent surface drift
    _allowed_validation = _load_validation_allowlist()
    validation_filtered = {
        k: bool(v) for k, v in validation_flags.items() if k in _allowed_validation
    }

    evaluation_report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": current_run_id,
        "meta": meta,
        "auto": auto,
        "dataset": dataset_info,
        "edit": edit_metadata,
        "telemetry": telemetry,
        "baseline_ref": baseline_ref,
        "invariants": invariants,
        "spectral": spectral,
        "rmt": rmt,
        "variance": variance,
        "structure": structure,
        "policies": policies,
        "resolved_policy": resolved_policy,
        "policy_provenance": policy_provenance,
        "provenance": provenance,
        "plugins": plugin_provenance,
        "edit_name": (report.get("edit", {}) or {}).get(
            "name", "unknown"
        ),  # Include edit name for rendering
        "artifacts": artifacts_payload,
        "validation": validation_filtered,
        "guard_overhead": guard_overhead_section,
        "primary_metric_tail": pm_tail_result,
    }

    # Record tiny-relax provenance explicitly when active.
    if tiny_relax:
        try:
            evaluation_report.setdefault("auto", {})["tiny_relax"] = True
            prov = evaluation_report.setdefault("provenance", {})
            flags = prov.setdefault("flags", [])
            if "tiny_relax" not in flags:
                flags.append("tiny_relax")
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pass

    # Compute PM-aware quality overhead when both snapshots are present
    try:
        pm_kind_hint = None
        try:
            pm_try = (
                report.get("metrics", {}).get("primary_metric")
                if isinstance(report.get("metrics"), dict)
                else None
            )
            if isinstance(pm_try, dict):
                pm_kind_hint = pm_try.get("kind")
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pm_kind_hint = None
        qo = _compute_quality_overhead_from_guard(raw_guard_ctx, pm_kind_hint)
        if (
            isinstance(qo, dict)
            and "value" in qo
            and math.isfinite(float(qo.get("value", float("nan"))))
        ):
            evaluation_report["quality_overhead"] = qo
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    try:
        _propagate_pairing_stats(evaluation_report, ppl_analysis)
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    # Attach policy/version digest object (thresholds/floors + key knobs)
    try:
        cur_tier = str(auto.get("tier", "balanced")).lower()
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        cur_tier = "balanced"
    thresholds_payload = _compute_thresholds_payload(cur_tier, resolved_policy)
    thresholds_hash = _compute_thresholds_hash(thresholds_payload)
    # Baseline tier for change note (best-effort)
    base_tier = None
    try:
        # Prefer raw baseline RunReport (if provided)
        if isinstance(baseline_raw, dict):
            bm = baseline_raw.get("meta")
            if isinstance(bm, dict):
                ba = bm.get("auto")
                if isinstance(ba, dict) and ba.get("tier"):
                    base_tier = str(ba.get("tier")).lower()
        # Fallback to normalized (usually lacks meta)
        if base_tier is None and isinstance(baseline_normalized, dict):
            base_meta = baseline_normalized.get("meta")
            if isinstance(base_meta, dict):
                base_auto = base_meta.get("auto")
                if isinstance(base_auto, dict) and base_auto.get("tier"):
                    base_tier = str(base_auto.get("tier")).lower()
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        base_tier = None
    baseline_payload = _compute_thresholds_payload(
        base_tier or cur_tier, resolved_policy
    )
    baseline_hash = _compute_thresholds_hash(baseline_payload)
    changed = bool(
        (base_tier is not None and base_tier != cur_tier)
        or (baseline_hash != thresholds_hash)
    )

    # Hysteresis knobs snapshot (policy-resolved)
    metrics_policy = (
        resolved_policy.get("metrics", {}) if isinstance(resolved_policy, dict) else {}
    )
    if not isinstance(metrics_policy, dict):
        metrics_policy = {}
    ppl_hys = 0.0
    acc_hys = 0.0
    try:
        ppl_hys = float(
            (metrics_policy.get("pm_ratio") or {}).get("hysteresis_ratio", 0.0) or 0.0
        )
        acc_hys = float(
            (metrics_policy.get("accuracy") or {}).get("hysteresis_delta_pp", 0.0)
            or 0.0
        )
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass
    min_effective = float(
        (resolved_policy.get("variance") or {}).get("min_effect_lognll", 0.0) or 0.0
    )

    evaluation_report["policy_digest"] = {
        "policy_version": POLICY_VERSION,
        "tier_policy_name": cur_tier,
        "thresholds_hash": thresholds_hash,
        "hysteresis": {"ppl": ppl_hys, "accuracy_delta_pp": acc_hys},
        "min_effective": min_effective,
        "changed": changed,
    }

    # Optional: include secondary metrics (informational; non-gating)
    try:
        if isinstance(report.get("metrics"), dict):
            sec = report["metrics"].get("secondary_metrics")
            if isinstance(sec, list) and sec:
                sanitized: list[dict[str, Any]] = []
                for item in sec:
                    if isinstance(item, dict) and item.get("kind"):
                        payload: dict[str, Any] = {}
                        for key in (
                            "kind",
                            "preview",
                            "final",
                            "ratio_vs_baseline",
                            "unit",
                            "display_ci",
                            "ci",
                        ):
                            if key in item:
                                payload[key] = item[key]
                        sanitized.append(payload)
                if sanitized:
                    evaluation_report["secondary_metrics"] = sanitized
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    # Optional: classification subgroup analysis (informational)
    try:
        cls = (
            report.get("metrics", {}).get("classification")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        if isinstance(cls, dict):
            sub = cls.get("subgroups")
            # Expect pre-aggregated subgroup counts
            if isinstance(sub, dict) and all(k in sub for k in ("preview", "final")):
                prev = sub.get("preview", {})
                fin = sub.get("final", {})
                pc = prev.get("group_counts", {}) if isinstance(prev, dict) else {}
                pcc = prev.get("correct_counts", {}) if isinstance(prev, dict) else {}
                fc = fin.get("group_counts", {}) if isinstance(fin, dict) else {}
                fcc = fin.get("correct_counts", {}) if isinstance(fin, dict) else {}
                out: dict[str, Any] = {}
                labels = set(list(pc.keys()) + list(fc.keys()))
                for g in labels:
                    try:
                        nprev = float(pc.get(g, 0))
                        nfin = float(fc.get(g, 0))
                        acc_prev = (
                            float(pcc.get(g, 0)) / nprev if nprev > 0 else float("nan")
                        )
                        acc_fin = (
                            float(fcc.get(g, 0)) / nfin if nfin > 0 else float("nan")
                        )
                        delta_pp = (
                            (acc_fin - acc_prev) * 100.0
                            if (math.isfinite(acc_prev) and math.isfinite(acc_fin))
                            else float("nan")
                        )
                        out[str(g)] = {
                            "preview": acc_prev,
                            "final": acc_fin,
                            "delta_pp": delta_pp,
                            "n_preview": nprev,
                            "n_final": nfin,
                        }
                    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
                        continue
                if out:
                    evaluation_report["classification"] = {"subgroups": out}
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    # Compute System Overhead (latency/throughput) vs baseline when available
    try:

        def _extract_sys_metrics(container: dict[str, Any] | None) -> dict[str, float]:
            out: dict[str, float] = {}
            if not isinstance(container, dict):
                return out
            metrics = (
                container.get("metrics", {})
                if isinstance(container.get("metrics"), dict)
                else {}
            )
            # Edited report case: also check evaluation_report telemetry keys
            telem = telemetry if isinstance(telemetry, dict) else {}
            # Prefer explicit p50/p95 throughput keys if present
            for key in ("latency_ms_p50", "latency_ms_p95", "throughput_sps"):
                val = metrics.get(key)
                if isinstance(val, int | float) and math.isfinite(float(val)):
                    out[key] = float(val)
            # Fallbacks
            if "latency_ms_p50" not in out:
                val = metrics.get("latency_ms_per_tok") or telem.get(
                    "latency_ms_per_tok"
                )
                if isinstance(val, int | float) and math.isfinite(float(val)):
                    out["latency_ms_p50"] = float(val)
            if "throughput_sps" not in out:
                val = metrics.get("throughput_tok_per_s") or telem.get(
                    "throughput_tok_per_s"
                )
                if isinstance(val, int | float) and math.isfinite(float(val)):
                    out["throughput_sps"] = float(val)
            return out

        edited_sys = _extract_sys_metrics(report)
        base_sys = _extract_sys_metrics(
            baseline_raw if isinstance(baseline_raw, dict) else None
        )
        system_overhead: dict[str, Any] = {}
        for metric_key, edited_val in edited_sys.items():
            base_val = base_sys.get(metric_key)
            entry: dict[str, Any] = {"edited": edited_val}
            if isinstance(base_val, int | float) and math.isfinite(float(base_val)):
                entry["baseline"] = float(base_val)
                entry["delta"] = float(edited_val - base_val)
                try:
                    entry["ratio"] = (
                        float(edited_val / base_val) if base_val != 0 else float("nan")
                    )
                except NON_FATAL_EXCEPTIONS:  # pragma: no cover
                    entry["ratio"] = float("nan")
            system_overhead[metric_key] = entry
        if system_overhead:
            evaluation_report["system_overhead"] = system_overhead
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    # Attach/normalize primary metric block (moved to helper)
    from .primary_metric_utils import attach_primary_metric as _attach_pm

    _attach_pm(evaluation_report, report, baseline_raw, baseline_ref, ppl_analysis)
    try:
        if isinstance(pm_drift_band, dict) and pm_drift_band:
            pm_block = evaluation_report.get("primary_metric")
            if isinstance(pm_block, dict):
                pm_block.setdefault("drift_band", dict(pm_drift_band))
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass
    _enforce_display_ci_alignment(
        ratio_ci_source,
        evaluation_report.get("primary_metric"),
        logloss_delta_ci,
        window_plan_profile,
    )

    # Ensure primary_metric has display_ci populated for schema invariants
    try:
        pm = (
            evaluation_report.get("primary_metric", {})
            if isinstance(evaluation_report.get("primary_metric"), dict)
            else None
        )
        if isinstance(pm, dict) and pm:
            # Prefer existing bounds; otherwise collapse to point estimate
            disp = pm.get("display_ci")
            if not (
                isinstance(disp, list | tuple)
                and len(disp) == 2
                and all(isinstance(x, int | float) for x in disp)
            ):
                point = None
                for key in ("ratio_vs_baseline", "final", "preview"):
                    val = pm.get(key)
                    if isinstance(val, int | float) and math.isfinite(float(val)):
                        point = float(val)
                        break
                if isinstance(point, float):
                    pm["display_ci"] = [point, point]
                else:
                    # As last resort, emit a degenerate [1.0, 1.0] to satisfy schema invariants
                    pm["display_ci"] = [1.0, 1.0]
                    pm.setdefault("estimated", True)
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    # Emit optional one-line telemetry summary (opt-in via INVARLOCK_TELEMETRY=1).
    # This runs after primary_metric attachment so the summary can include display_ci/width.
    try:
        kind = None
        pm_try = (
            report.get("metrics", {}).get("primary_metric")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        if isinstance(pm_try, dict):
            kind = pm_try.get("kind")
        if not kind:
            kind = "ppl"
        windows_cfg = (
            evaluation_report.get("dataset", {}).get("windows", {})
            if isinstance(evaluation_report.get("dataset"), dict)
            else {}
        )
        n_prev = windows_cfg.get("preview")
        n_fin = windows_cfg.get("final")
        tokens_total = None
        try:
            tokens_total = (
                evaluation_report.get("dataset", {}).get("hash", {}).get("total_tokens")
            )
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            tokens_total = None
        # CI interval
        ci_lo = None
        ci_hi = None
        ratio = None
        pmc = evaluation_report.get("primary_metric", {})
        rci = pmc.get("display_ci") or pmc.get("ci")
        if isinstance(rci, tuple | list) and len(rci) == 2:
            ci_lo, ci_hi = rci[0], rci[1]
        ratio = pmc.get("ratio_vs_baseline")
        ci_w = None
        try:
            if isinstance(ci_lo, int | float) and isinstance(ci_hi, int | float):
                ci_w = float(ci_hi) - float(ci_lo)
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            ci_w = None
        # Gate outcome
        val = evaluation_report.get("validation", {})
        gate_ok = None
        try:
            gate_ok = bool(val.get("primary_metric_acceptable"))
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            gate_ok = None
        # Build line
        parts = [
            f"run_id={current_run_id}",
            f"metric={kind}",
            f"nprev={n_prev}",
            f"nfinal={n_fin}",
            f"tokens={tokens_total}",
        ]
        try:
            split = (evaluation_report.get("provenance", {}) or {}).get("dataset_split")
            if not split:
                split = (report.get("provenance", {}) or {}).get("dataset_split")
            sf = (evaluation_report.get("provenance", {}) or {}).get("split_fallback")
            if sf is None:
                sf = (report.get("provenance", {}) or {}).get("split_fallback")
            if split:
                parts.append(f"split={split}{'*' if sf else ''}")
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pass
        if isinstance(ci_lo, int | float) and isinstance(ci_hi, int | float):
            parts.append(f"ci={ci_lo:.3f}-{ci_hi:.3f}")
            if isinstance(ci_w, int | float):
                parts.append(f"width={ci_w:.3f}")
        if isinstance(ratio, int | float):
            parts.append(f"ratio={float(ratio):.3f}")
        if isinstance(gate_ok, bool):
            parts.append(f"gate={'pass' if gate_ok else 'fail'}")
        summary_line = "INVARLOCK_TELEMETRY " + " ".join(parts)
        evaluation_report.setdefault("telemetry", {})["summary_line"] = summary_line
        if str(os.environ.get("INVARLOCK_TELEMETRY", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            print(summary_line)
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    # Attach confidence label (non-gating)
    try:
        evaluation_report["confidence"] = _compute_confidence_label(evaluation_report)
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    return evaluation_report
