from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from .bootstrap import (
    compute_logloss_ci,
    compute_paired_delta_log_ci,
    logspace_to_ratio_ci,
)
from .runner_eval_windows import compute_slice_summary, resolve_limit, slice_calibration
from .runner_latency import measure_latency
from .runner_pairing import (
    BOOTSTRAP_COVERAGE_REQUIREMENTS,
    assess_bootstrap_coverage,
    compute_window_pairing_metrics,
)
from .types import LogLevel


def compute_real_metrics(
    runner: Any,
    model: Any,
    calibration_data: Any,
    adapter: Any,
    preview_n: int | None = None,
    final_n: int | None = None,
    config: Any | None = None,
    *,
    compute_paired_delta_log_ci_fn: Any = compute_paired_delta_log_ci,
    logspace_to_ratio_ci_fn: Any = logspace_to_ratio_ci,
    coverage_requirements: Any = BOOTSTRAP_COVERAGE_REQUIREMENTS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute evaluation metrics from calibration data."""
    import psutil  # type: ignore[import-untyped]
    import torch

    _ = adapter
    model.eval()

    if os.environ.get("INVARLOCK_DEBUG_TRACE"):
        runner._log_event(
            "eval",
            "real_metrics_snapshot",
            LogLevel.DEBUG,
            {
                "preview_n": preview_n,
                "final_n": final_n,
                "calibration_len": (
                    len(calibration_data)
                    if hasattr(calibration_data, "__len__")
                    else "n/a"
                ),
            },
        )
    device = next(model.parameters()).device

    eval_device_override = os.environ.get("INVARLOCK_EVAL_DEVICE")
    if eval_device_override:
        override_device = torch.device(eval_device_override)
        if override_device != device:
            model.to(override_device)
            device = override_device

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024

    policy_flags = runner._resolve_policy_flags(config)
    allow_materialize = policy_flags["allow_calibration_materialize"]

    if not hasattr(calibration_data, "__len__"):
        if allow_materialize and hasattr(calibration_data, "__iter__"):
            calibration_data = list(calibration_data)
        else:
            raise ValueError(
                "Calibration data must define __len__ (or enable materialization "
                "via INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE or context.run.allow_calibration_materialize)."
            )

    total_available = (
        len(calibration_data) if hasattr(calibration_data, "__len__") else 0
    )
    if total_available == 0:
        raise ValueError("Calibration data is empty; cannot compute metrics.")

    if preview_n is None:
        preview_n = max(total_available // 2, 1)
    if final_n is None:
        final_n = preview_n

    preview_n = max(int(preview_n), 0)
    final_n = max(int(final_n), 0)
    max_single_arm = max(preview_n, final_n)
    if max_single_arm <= 0:
        raise ValueError("preview_n and final_n cannot both be zero.")

    if max_single_arm > total_available:
        runner._log_event(
            "eval",
            "data_scaled",
            LogLevel.WARNING,
            {
                "requested_preview": preview_n,
                "requested_final": final_n,
                "available": total_available,
            },
        )
        preview_n = min(preview_n, total_available)
        final_n = min(final_n, total_available)

    requested_preview = preview_n
    requested_final = final_n
    total_needed = preview_n + final_n
    if total_needed > total_available:
        runner._log_event(
            "eval",
            "window_shortage",
            LogLevel.WARNING,
            {
                "requested_preview": preview_n,
                "requested_final": final_n,
                "available": total_available,
            },
        )

    preview_n = min(preview_n, total_available)
    final_start = preview_n
    remaining = max(total_available - preview_n, 0)
    if final_n > remaining:
        runner._log_event(
            "eval",
            "final_window_shortage",
            LogLevel.WARNING,
            {
                "requested_final": final_n,
                "available_after_preview": remaining,
                "requested_preview": requested_preview,
                "requested_final_original": requested_final,
            },
        )
        final_n = remaining

    preview_data, calibration_data = slice_calibration(
        calibration_data,
        start=0,
        count=preview_n,
        allow_materialize=allow_materialize,
    )
    final_data, calibration_data = slice_calibration(
        calibration_data,
        start=final_start,
        count=final_n,
        allow_materialize=allow_materialize,
    )

    eval_context: dict[str, Any] = {}
    if config and isinstance(config.context, dict):
        eval_context = config.context.get("eval", {}) or {}

    loss_cfg = eval_context.get("loss", {}) if isinstance(eval_context, dict) else {}
    resolved_loss_mode = str(
        loss_cfg.get("resolved_type") or loss_cfg.get("type") or ""
    ).lower()
    bootstrap_cfg = eval_context.get("bootstrap", {}) or {}
    bootstrap_enabled = bool(bootstrap_cfg.get("enabled", True))
    bootstrap_method = str(bootstrap_cfg.get("method", "bca_paired_delta_log")).lower()
    bootstrap_replicates = int(
        bootstrap_cfg.get("replicates", bootstrap_cfg.get("n", 1000) or 1000)
    )
    bootstrap_alpha = float(bootstrap_cfg.get("alpha", 0.05) or 0.05)
    bootstrap_seed_cfg = bootstrap_cfg.get("seed")
    ci_band = float(bootstrap_cfg.get("ci_band", 0.10) or 0.10)

    if bootstrap_method == "percentile":
        single_method = delta_method = "percentile"
    elif bootstrap_method == "bca_paired_delta_log":
        single_method = delta_method = "bca"
    else:
        single_method = bootstrap_method
        delta_method = bootstrap_method

    dataset_seed = None
    profile_label = ""
    pairing_context: dict[str, Any] = {}
    if config and isinstance(config.context, dict):
        dataset_cfg = config.context.get("dataset", {})
        if isinstance(dataset_cfg, dict):
            dataset_seed = dataset_cfg.get("seed")
        profile_label = str(config.context.get("profile", "")).lower()
        pairing_context = config.context.get("pairing_baseline", {}) or {}

    bootstrap_seed = (
        bootstrap_seed_cfg if bootstrap_seed_cfg is not None else dataset_seed
    )
    eval_error: dict[str, Any] | None = None
    try:
        bootstrap_seed = int(bootstrap_seed) if bootstrap_seed is not None else 0
    except (TypeError, ValueError):
        bootstrap_seed = 0

    if bootstrap_replicates <= 0:
        bootstrap_enabled = False
    if not (0.0 < bootstrap_alpha < 1.0):
        bootstrap_alpha = 0.05

    pm_preview = 50.0
    pm_final = 50.0
    ratio_ci: tuple[float, float] = (1.0, 1.0)
    preview_log_ci: tuple[float, float] = (math.log(pm_preview), math.log(pm_preview))
    final_log_ci: tuple[float, float] = (math.log(pm_final), math.log(pm_final))
    delta_log_ci: tuple[float, float] = (0.0, 0.0)
    preview_mean_log = math.log(pm_preview)
    final_mean_log = math.log(pm_final)
    delta_mean_log = 0.0
    preview_log_losses: list[float] = []
    final_log_losses: list[float] = []
    preview_tokens_ct = 0
    final_tokens_ct = 0
    preview_batches_ct = 0
    final_batches_ct = 0
    window_overlap_fraction = 0.0
    window_match_fraction = 1.0
    pairing_reason = None
    preview_pair_stats = {"matched": 0, "expected": 0}
    final_pair_stats = {"matched": 0, "expected": 0}
    paired_windows_attempted = 0
    preview_window_ids: list[int] = []
    final_window_ids: list[int] = []
    preview_tokens: list[list[int]] = []
    final_tokens: list[list[int]] = []
    preview_limit = min(preview_n, len(preview_data)) if preview_data else 0
    final_limit = min(final_n, len(final_data)) if final_data else 0
    preview_actual_tokens_ct = int(preview_tokens_ct)
    final_actual_tokens_ct = int(final_tokens_ct)
    preview_masked_total = int(preview_tokens_ct)
    final_masked_total = int(final_tokens_ct)
    preview_token_counts: list[int] = []
    final_token_counts: list[int] = []
    preview_attention_masks: list[list[int]] = []
    final_attention_masks: list[list[int]] = []
    preview_mask_counts: list[int] = []
    final_mask_counts: list[int] = []
    preview_labels: list[list[int]] = []
    final_labels: list[list[int]] = []
    preview_actual_token_counts: list[int] = []
    final_actual_token_counts: list[int] = []
    degenerate_delta = False
    degenerate_reason: str | None = None
    bootstrap_info = {
        "enabled": bool(bootstrap_enabled),
        "method": bootstrap_method,
        "alpha": float(bootstrap_alpha),
        "replicates": int(bootstrap_replicates),
        "seed": int(bootstrap_seed),
        "ci_band": float(ci_band),
    }
    delta_samples: list[float] = []
    delta_weights: list[float] = []
    pm_invalid = False
    degraded_reason: str | None = None

    try:
        preview_limit = resolve_limit(preview_data, preview_n)
        final_limit = resolve_limit(final_data, final_n)

        preview_summary, preview_error = compute_slice_summary(
            runner,
            model,
            preview_data,
            max_batches=preview_limit,
            start_idx=0,
            device=device,
            resolved_loss_mode=resolved_loss_mode,
        )
        final_summary, final_error = compute_slice_summary(
            runner,
            model,
            final_data,
            max_batches=final_limit,
            start_idx=preview_summary["num_batches"],
            device=device,
            resolved_loss_mode=resolved_loss_mode,
        )
        eval_error = preview_error or final_error

        preview_raw_losses = preview_summary["log_losses"]
        final_raw_losses = final_summary["log_losses"]
        try:
            paired_windows_attempted = min(
                len(preview_raw_losses), len(final_raw_losses)
            )
        except Exception:
            paired_windows_attempted = 0

        preview_log_losses = [
            float(loss) for loss in preview_raw_losses if math.isfinite(loss)
        ]
        final_log_losses = [
            float(loss) for loss in final_raw_losses if math.isfinite(loss)
        ]
        if len(preview_log_losses) != len(preview_raw_losses):
            runner._log_event(
                "eval",
                "non_finite_preview_losses_filtered",
                LogLevel.WARNING,
                {
                    "total": len(preview_raw_losses),
                    "filtered": len(preview_raw_losses) - len(preview_log_losses),
                },
            )
        if len(final_log_losses) != len(final_raw_losses):
            runner._log_event(
                "eval",
                "non_finite_final_losses_filtered",
                LogLevel.WARNING,
                {
                    "total": len(final_raw_losses),
                    "filtered": len(final_raw_losses) - len(final_log_losses),
                },
            )

        preview_tokens_ct = preview_summary["total_tokens"]
        final_tokens_ct = final_summary["total_tokens"]
        preview_batches_ct = preview_summary["num_batches"]
        final_batches_ct = final_summary["num_batches"]
        preview_window_ids = list(preview_summary["window_ids"])
        final_window_ids = list(final_summary["window_ids"])
        preview_tokens = list(preview_summary["tokens"])
        final_tokens = list(final_summary["tokens"])
        preview_token_counts = list(preview_summary.get("window_token_counts", []))
        final_token_counts = list(final_summary.get("window_token_counts", []))
        preview_attention_masks = list(preview_summary.get("attention_masks", []))
        final_attention_masks = list(final_summary.get("attention_masks", []))
        preview_mask_counts = list(preview_summary.get("masked_token_counts", []))
        final_mask_counts = list(final_summary.get("masked_token_counts", []))
        preview_labels = list(preview_summary.get("labels", []))
        final_labels = list(final_summary.get("labels", []))
        preview_actual_token_counts = list(
            preview_summary.get("actual_token_counts", [])
        )
        final_actual_token_counts = list(final_summary.get("actual_token_counts", []))
        preview_actual_tokens_ct = int(
            preview_summary.get("actual_total_tokens", preview_tokens_ct)
        )
        final_actual_tokens_ct = int(
            final_summary.get("actual_total_tokens", final_tokens_ct)
        )
        preview_masked_total = (
            sum(preview_mask_counts) if preview_mask_counts else int(preview_tokens_ct)
        )
        final_masked_total = (
            sum(final_mask_counts) if final_mask_counts else int(final_tokens_ct)
        )
        preview_weighted_loss = float(preview_summary.get("weighted_log_loss", 0.0))
        final_weighted_loss = float(final_summary.get("weighted_log_loss", 0.0))

        if preview_tokens_ct > 0:
            preview_mean_log = float(preview_weighted_loss / preview_tokens_ct)
            pm_preview = math.exp(preview_mean_log)
        elif preview_log_losses:
            preview_mean_log = float(np.mean(preview_log_losses))
            pm_preview = math.exp(preview_mean_log)
        else:
            pm_preview = preview_summary["ppl"]
            if not math.isfinite(pm_preview) or pm_preview <= 0:
                pm_preview = 50.0
            preview_mean_log = math.log(pm_preview)

        if final_tokens_ct > 0:
            final_mean_log = float(final_weighted_loss / final_tokens_ct)
            pm_final = math.exp(final_mean_log)
        elif final_log_losses:
            final_mean_log = float(np.mean(final_log_losses))
            pm_final = math.exp(final_mean_log)
        else:
            pm_final = final_summary["ppl"]
            if not math.isfinite(pm_final) or pm_final <= 0:
                pm_final = 50.0
            final_mean_log = math.log(pm_final)

        delta_mean_log = final_mean_log - preview_mean_log
        ppl_ratio = math.exp(delta_mean_log)

        try:
            if not (math.isfinite(delta_mean_log) and math.isfinite(ppl_ratio)):
                raise RuntimeError("non_finite_primary_metric")
            expected_ratio = math.exp(delta_mean_log)
            if abs(ppl_ratio - expected_ratio) > 1e-6:
                raise RuntimeError("primary_metric_ratio_mismatch")
        except Exception as exc:
            pm_invalid = True
            runner._log_event(
                "eval",
                "primary_metric_invalid",
                LogLevel.WARNING,
                {
                    "pm_preview": float(pm_preview),
                    "pm_final": float(pm_final),
                    "delta_mean_log": float(delta_mean_log),
                    "pm_ratio": float(ppl_ratio),
                    "error": str(exc),
                },
            )

        if bootstrap_enabled and preview_log_losses:
            preview_log_ci = compute_logloss_ci(
                preview_log_losses,
                method=single_method,
                replicates=bootstrap_replicates,
                alpha=bootstrap_alpha,
                seed=bootstrap_seed + 7,
            )
        else:
            preview_log_ci = (preview_mean_log, preview_mean_log)

        if bootstrap_enabled and final_log_losses:
            final_log_ci = compute_logloss_ci(
                final_log_losses,
                method=single_method,
                replicates=bootstrap_replicates,
                alpha=bootstrap_alpha,
                seed=bootstrap_seed + 13,
            )
        else:
            final_log_ci = (final_mean_log, final_mean_log)

        paired_weights: list[float] | None = None
        if preview_token_counts:
            paired_weights = [float(max(weight, 0)) for weight in preview_token_counts]
        elif final_token_counts:
            paired_weights = [float(max(weight, 0)) for weight in final_token_counts]

        if bootstrap_enabled and final_log_losses and preview_log_losses:
            delta_log_ci = compute_paired_delta_log_ci_fn(
                final_log_losses,
                preview_log_losses,
                weights=paired_weights,
                method=delta_method,
                replicates=bootstrap_replicates,
                alpha=bootstrap_alpha,
                seed=bootstrap_seed + 97,
            )
            ratio_ci = logspace_to_ratio_ci_fn(delta_log_ci)
            expected_ratio_ci = tuple(math.exp(bound) for bound in delta_log_ci)
            if any(
                abs(ratio_bound - expected_bound) > 1e-6
                for ratio_bound, expected_bound in zip(
                    ratio_ci, expected_ratio_ci, strict=False
                )
            ):
                pm_invalid = True
                runner._log_event(
                    "eval",
                    "ratio_ci_inconsistent",
                    LogLevel.WARNING,
                    {
                        "ratio_ci": ratio_ci,
                        "expected_ratio_ci": expected_ratio_ci,
                    },
                )
                ratio_ci = (float(expected_ratio_ci[0]), float(expected_ratio_ci[1]))
        else:
            delta_log_ci = (delta_mean_log, delta_mean_log)
            ratio_ci = (ppl_ratio, ppl_ratio)

        if final_log_losses and preview_log_losses:
            limit = min(len(final_log_losses), len(preview_log_losses))
            if limit:
                delta_samples = [
                    final_log_losses[index] - preview_log_losses[index]
                    for index in range(limit)
                ]
                if preview_token_counts and len(preview_token_counts) >= limit:
                    delta_weights = [
                        float(max(preview_token_counts[index], 1))
                        for index in range(limit)
                    ]
                elif final_token_counts and len(final_token_counts) >= limit:
                    delta_weights = [
                        float(max(final_token_counts[index], 1))
                        for index in range(limit)
                    ]

        if len(delta_samples) < 2:
            degenerate_delta = True
            degenerate_reason = "no_pairs" if len(delta_samples) == 0 else "single_pair"
        elif np.allclose(delta_samples, delta_samples[0]):
            degenerate_delta = True
            degenerate_reason = "no_variation"

        if degenerate_delta:
            pm_invalid = True
            runner._log_event(
                "eval",
                "degenerate_delta_samples",
                LogLevel.WARNING,
                {"reason": degenerate_reason, "sample_count": len(delta_samples)},
            )

        needs_pm_fallback = (not math.isfinite(pm_preview)) or (
            not math.isfinite(pm_final)
        )
        needs_delta_fallback = (not math.isfinite(delta_mean_log)) or (
            not math.isfinite(ppl_ratio)
        )

        if needs_pm_fallback:
            degraded_reason = "non_finite_pm"
        elif needs_delta_fallback:
            degraded_reason = "non_finite_delta"
        elif degenerate_reason:
            degraded_reason = f"degenerate_delta:{degenerate_reason}"
        elif pm_invalid:
            degraded_reason = "primary_metric_invalid"

        if needs_pm_fallback or needs_delta_fallback:
            pm_invalid = True
            pm_fallback = (
                pm_preview if math.isfinite(pm_preview) and pm_preview > 0 else pm_final
            )
            if not (math.isfinite(pm_fallback) and pm_fallback > 0):
                pm_fallback = 1.0
            if needs_pm_fallback:
                pm_preview = (
                    pm_preview
                    if math.isfinite(pm_preview) and pm_preview > 0
                    else pm_fallback
                )
                pm_final = (
                    pm_final
                    if math.isfinite(pm_final) and pm_final > 0
                    else pm_fallback
                )
            if needs_delta_fallback:
                if not math.isfinite(delta_mean_log):
                    delta_mean_log = 0.0
                if not math.isfinite(ppl_ratio):
                    ppl_ratio = 1.0

        pairing_metrics = compute_window_pairing_metrics(
            preview_window_ids=preview_window_ids,
            preview_tokens=preview_tokens,
            final_window_ids=final_window_ids,
            final_tokens=final_tokens,
            pairing_context=pairing_context
            if isinstance(pairing_context, dict)
            else None,
            config_context=config.context
            if config and isinstance(config.context, dict)
            else None,
            preview_batches=preview_batches_ct,
            final_batches=final_batches_ct,
        )
        preview_pair_stats = pairing_metrics["preview"]
        final_pair_stats = pairing_metrics["final"]
        window_match_fraction = float(pairing_metrics["match_fraction"])
        window_overlap_fraction = float(pairing_metrics["overlap_fraction"])
        duplicate_fraction = float(pairing_metrics["duplicate_fraction"])
        count_mismatch = bool(pairing_metrics["count_mismatch"])
        pairing_reason = pairing_metrics["reason"]

        if pairing_context and window_match_fraction < 0.999999:
            runner._log_event(
                "eval",
                "window_pairing_mismatch",
                LogLevel.ERROR,
                {
                    "match_fraction": window_match_fraction,
                    "overlap_fraction": window_overlap_fraction,
                    "reason": pairing_reason,
                    "preview": preview_pair_stats,
                    "final": final_pair_stats,
                },
            )
        if window_overlap_fraction > 0.0 and pairing_context:
            runner._log_event(
                "eval",
                "window_overlap_warning",
                LogLevel.WARNING,
                {
                    "overlap_fraction": window_overlap_fraction,
                    "duplicate_fraction": duplicate_fraction,
                    "match_fraction": window_match_fraction,
                    "preview": preview_pair_stats,
                    "final": final_pair_stats,
                },
            )

        if pairing_context and profile_label in {"ci", "release"}:
            if window_match_fraction < 0.999999:
                raise RuntimeError(
                    f"Window pairing mismatch detected (fraction={window_match_fraction:.3f}, reason={pairing_reason})"
                )
            if window_overlap_fraction > 0.0:
                raise RuntimeError(
                    f"Window overlap detected (overlap_fraction={window_overlap_fraction:.3f})"
                )
            if count_mismatch:
                raise RuntimeError(
                    f"Window count mismatch detected (preview={preview_batches_ct}, final={final_batches_ct})"
                )

        tier = "balanced"
        if config and isinstance(config.context, dict):
            auto_section = config.context.get("auto", {})
            if isinstance(auto_section, dict):
                tier = str(auto_section.get("tier", tier)).lower()

        coverage_summary = assess_bootstrap_coverage(
            tier=tier,
            preview_batches=preview_batches_ct,
            final_batches=final_batches_ct,
            bootstrap_enabled=bool(bootstrap_enabled),
            bootstrap_replicates=int(bootstrap_replicates),
            requirements=coverage_requirements,
        )
        preview_required = int(coverage_summary["preview_required"])
        final_required = int(coverage_summary["final_required"])
        replicates_required = int(coverage_summary["replicates_required"])
        preview_ok = bool(coverage_summary["preview_ok"])
        final_ok = bool(coverage_summary["final_ok"])
        replicates_ok = bool(coverage_summary["replicates_ok"])

        if not (preview_ok and final_ok and replicates_ok):
            runner._log_event(
                "eval",
                "bootstrap_coverage_warning",
                LogLevel.WARNING,
                {
                    "tier": tier,
                    "preview_used": preview_batches_ct,
                    "preview_required": preview_required,
                    "final_used": final_batches_ct,
                    "final_required": final_required,
                    "replicates_used": bootstrap_replicates,
                    "replicates_required": replicates_required,
                },
            )
            if pairing_context and profile_label in {"ci", "release"}:
                from invarlock.core.exceptions import InvarlockError

                raise InvarlockError(
                    code="E005",
                    message=(
                        "INSUFFICIENT-SAMPLE: bootstrap coverage below policy floors in CI/Release"
                    ),
                )

        bootstrap_info.update(
            {
                "enabled": bool(bootstrap_enabled),
                "method": bootstrap_method,
                "alpha": float(bootstrap_alpha),
                "replicates": int(bootstrap_replicates),
                "seed": int(bootstrap_seed),
                "ci_band": float(ci_band),
                "window_duplicate_fraction": float(duplicate_fraction),
                "window_match_fraction": float(window_match_fraction),
                "coverage": coverage_summary["coverage"],
            }
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        runner._log_event(
            "eval",
            "error",
            LogLevel.ERROR,
            {"message": f"Primary-metric computation failed: {exc}"},
        )
        eval_error = {"type": type(exc).__name__, "message": str(exc)}

    latency_ms_per_tok = measure_latency(
        model, preview_data[:1] if preview_data else final_data[:1], device
    )
    current_memory = process.memory_info().rss / 1024 / 1024
    peak_memory = max(initial_memory, current_memory)

    eval_samples = 0
    total_tokens = 0
    masked_total_tokens = 0
    try:
        eval_samples = int(preview_batches_ct) + int(final_batches_ct)
        total_tokens = int(preview_actual_tokens_ct) + int(final_actual_tokens_ct)
        masked_total_tokens = int(preview_masked_total) + int(final_masked_total)
    except Exception:
        pass

    paired_windows_count = (
        paired_windows_attempted if paired_windows_attempted else len(delta_samples)
    )
    unweighted_delta_mean = (
        float(np.mean(delta_samples)) if delta_samples else float(delta_mean_log)
    )
    preview_weighted_delta_mean: float | None = None
    if delta_weights:
        total_weight = float(sum(delta_weights))
        if total_weight > 0.0:
            preview_weighted_delta_mean = float(
                np.dot(delta_samples, delta_weights) / total_weight
            )
    paired_delta_mean = float(delta_mean_log)
    paired_delta_std = (
        float(np.std(delta_samples, ddof=1)) if len(delta_samples) > 1 else 0.0
    )
    paired_delta_min = float(min(delta_samples)) if delta_samples else None
    paired_delta_max = float(max(delta_samples)) if delta_samples else None

    pm_kind = "ppl_causal"
    if resolved_loss_mode == "mlm":
        pm_kind = "ppl_mlm"
    elif resolved_loss_mode in {"seq2seq", "s2s", "t5"}:
        pm_kind = "ppl_seq2seq"

    metrics = {
        "primary_metric": {
            "kind": pm_kind,
            "preview": float(pm_preview) if math.isfinite(pm_preview) else None,
            "final": float(pm_final) if math.isfinite(pm_final) else None,
            "invalid": bool(pm_invalid),
            "degraded": bool(pm_invalid or degraded_reason),
            "degraded_reason": degraded_reason,
        },
        "logloss_preview": float(preview_mean_log),
        "logloss_final": float(final_mean_log),
        "logloss_delta": float(delta_mean_log),
        "logloss_preview_ci": tuple(map(float, preview_log_ci)),
        "logloss_final_ci": tuple(map(float, final_log_ci)),
        "logloss_delta_ci": tuple(map(float, delta_log_ci)),
        "latency_ms_per_tok": latency_ms_per_tok,
        "memory_mb_peak": peak_memory,
        "eval_samples": eval_samples,
        "total_tokens": total_tokens,
        "preview_total_tokens": int(preview_actual_tokens_ct),
        "final_total_tokens": int(final_actual_tokens_ct),
        "masked_tokens_total": masked_total_tokens,
        "masked_tokens_preview": int(preview_masked_total),
        "masked_tokens_final": int(final_masked_total),
        "reduction": {
            "mode": "token_mean",
            "implementation": "huggingface_cross_entropy",
        },
        "window_overlap_fraction": float(window_overlap_fraction),
        "window_match_fraction": float(window_match_fraction),
        "window_pairing_reason": pairing_reason,
        "window_pairing_preview": {
            "matched": preview_pair_stats["matched"],
            "expected": preview_pair_stats["expected"],
            "reason": preview_pair_stats.get("reason"),
        },
        "window_pairing_final": {
            "matched": final_pair_stats["matched"],
            "expected": final_pair_stats["expected"],
            "reason": final_pair_stats.get("reason"),
        },
        "bootstrap": bootstrap_info,
        "paired_windows": paired_windows_count,
        "paired_delta_summary": {
            "mean": paired_delta_mean,
            "mean_unweighted": unweighted_delta_mean,
            "mean_preview_weighted": (
                preview_weighted_delta_mean
                if preview_weighted_delta_mean is not None
                else unweighted_delta_mean
            ),
            "std": paired_delta_std,
            "min": paired_delta_min,
            "max": paired_delta_max,
            "degenerate": degenerate_delta,
            "degenerate_reason": degenerate_reason,
        },
    }
    if eval_error:
        metrics["eval_error"] = eval_error

    eval_windows = {
        "preview": {
            "window_ids": preview_window_ids[:preview_limit],
            "logloss": list(preview_log_losses),
            "input_ids": preview_tokens,
            "attention_masks": preview_attention_masks,
            "token_counts": preview_token_counts,
            "masked_token_counts": preview_mask_counts,
            "actual_token_counts": preview_actual_token_counts,
            "labels": preview_labels,
        },
        "final": {
            "window_ids": final_window_ids[:final_limit],
            "logloss": list(final_log_losses),
            "input_ids": final_tokens,
            "attention_masks": final_attention_masks,
            "token_counts": final_token_counts,
            "masked_token_counts": final_mask_counts,
            "actual_token_counts": final_actual_token_counts,
            "labels": final_labels,
        },
    }
    return metrics, eval_windows


__all__ = ["compute_real_metrics"]
