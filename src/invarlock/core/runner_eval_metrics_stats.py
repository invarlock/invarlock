from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bootstrap import compute_logloss_ci
from .exceptions import InvarlockError
from .runner_pairing import assess_bootstrap_coverage, compute_window_pairing_metrics
from .types import LogLevel


@dataclass(frozen=True)
class _BootstrapDeltaResult:
    preview_log_ci: tuple[float, float]
    final_log_ci: tuple[float, float]
    delta_log_ci: tuple[float, float]
    ratio_ci: tuple[float, float]
    delta_samples: list[float]
    delta_weights: list[float]
    degenerate_delta: bool
    degenerate_reason: str | None
    pm_invalid: bool
    degraded_reason: str | None


@dataclass(frozen=True)
class _PairingCoverageResult:
    preview_pair_stats: dict[str, Any]
    final_pair_stats: dict[str, Any]
    window_overlap_fraction: float
    window_match_fraction: float
    pairing_reason: str | None
    bootstrap_info: dict[str, Any]


def _compute_bootstrap_delta_stats(
    runner: Any,
    runtime: Any,
    slices: Any,
    *,
    compute_paired_delta_log_ci_fn: Any,
    logspace_to_ratio_ci_fn: Any,
) -> _BootstrapDeltaResult:
    preview_log_ci = (slices.preview_mean_log, slices.preview_mean_log)
    final_log_ci = (slices.final_mean_log, slices.final_mean_log)
    delta_log_ci = (slices.delta_mean_log, slices.delta_mean_log)
    ratio_ci = (slices.ppl_ratio, slices.ppl_ratio)
    delta_samples: list[float] = []
    delta_weights: list[float] = []
    pm_invalid = bool(slices.pm_invalid)
    degenerate_delta = False
    degenerate_reason: str | None = None

    if runtime.bootstrap_enabled and slices.preview_log_losses:
        preview_log_ci = compute_logloss_ci(
            slices.preview_log_losses,
            method=runtime.single_method,
            replicates=runtime.bootstrap_replicates,
            alpha=runtime.bootstrap_alpha,
            seed=runtime.bootstrap_seed + 7,
        )
    if runtime.bootstrap_enabled and slices.final_log_losses:
        final_log_ci = compute_logloss_ci(
            slices.final_log_losses,
            method=runtime.single_method,
            replicates=runtime.bootstrap_replicates,
            alpha=runtime.bootstrap_alpha,
            seed=runtime.bootstrap_seed + 13,
        )

    paired_weights: list[float] | None = None
    if slices.preview_token_counts:
        paired_weights = [
            float(max(weight, 0)) for weight in slices.preview_token_counts
        ]
    elif slices.final_token_counts:
        paired_weights = [float(max(weight, 0)) for weight in slices.final_token_counts]

    if (
        runtime.bootstrap_enabled
        and slices.final_log_losses
        and slices.preview_log_losses
    ):
        try:
            delta_log_ci = compute_paired_delta_log_ci_fn(
                slices.final_log_losses,
                slices.preview_log_losses,
                weights=paired_weights,
                method=runtime.delta_method,
                replicates=runtime.bootstrap_replicates,
                alpha=runtime.bootstrap_alpha,
                seed=runtime.bootstrap_seed + 97,
                strict_lengths=True,
            )
        except ValueError as exc:
            pm_invalid = True
            runner._log_event(
                "eval",
                "paired_delta_strict_length_error",
                LogLevel.ERROR,
                {"reason": str(exc)},
            )
            delta_log_ci = compute_paired_delta_log_ci_fn(
                slices.final_log_losses,
                slices.preview_log_losses,
                weights=paired_weights,
                method=runtime.delta_method,
                replicates=runtime.bootstrap_replicates,
                alpha=runtime.bootstrap_alpha,
                seed=runtime.bootstrap_seed + 97,
                strict_lengths=False,
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

    if slices.final_log_losses and slices.preview_log_losses:
        limit = min(len(slices.final_log_losses), len(slices.preview_log_losses))
        if limit:
            delta_samples = [
                slices.final_log_losses[index] - slices.preview_log_losses[index]
                for index in range(limit)
            ]
            if (
                slices.preview_token_counts
                and len(slices.preview_token_counts) >= limit
            ):
                delta_weights = [
                    float(max(slices.preview_token_counts[index], 1))
                    for index in range(limit)
                ]
            elif slices.final_token_counts and len(slices.final_token_counts) >= limit:
                delta_weights = [
                    float(max(slices.final_token_counts[index], 1))
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

    degraded_reason: str | None = None
    needs_pm_fallback = (not math.isfinite(slices.pm_preview)) or (
        not math.isfinite(slices.pm_final)
    )
    needs_delta_fallback = (not math.isfinite(slices.delta_mean_log)) or (
        not math.isfinite(slices.ppl_ratio)
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

    return _BootstrapDeltaResult(
        preview_log_ci=(float(preview_log_ci[0]), float(preview_log_ci[1])),
        final_log_ci=(float(final_log_ci[0]), float(final_log_ci[1])),
        delta_log_ci=(float(delta_log_ci[0]), float(delta_log_ci[1])),
        ratio_ci=(float(ratio_ci[0]), float(ratio_ci[1])),
        delta_samples=delta_samples,
        delta_weights=delta_weights,
        degenerate_delta=degenerate_delta,
        degenerate_reason=degenerate_reason,
        pm_invalid=pm_invalid,
        degraded_reason=degraded_reason,
    )


def _evaluate_pairing_and_coverage(
    runner: Any,
    runtime: Any,
    slices: Any,
    *,
    config: Any | None,
    coverage_requirements: Any,
    compute_window_pairing_metrics_fn: Any = compute_window_pairing_metrics,
    assess_bootstrap_coverage_fn: Any = assess_bootstrap_coverage,
) -> _PairingCoverageResult:
    pairing_metrics = compute_window_pairing_metrics_fn(
        preview_window_ids=slices.preview_window_ids,
        preview_tokens=slices.preview_tokens,
        preview_labels=getattr(slices, "preview_labels", None),
        final_window_ids=slices.final_window_ids,
        final_tokens=slices.final_tokens,
        final_labels=getattr(slices, "final_labels", None),
        pairing_context=runtime.pairing_context
        if isinstance(runtime.pairing_context, dict)
        else None,
        config_context=config.context
        if config and isinstance(config.context, dict)
        else None,
        preview_batches=slices.preview_batches_ct,
        final_batches=slices.final_batches_ct,
    )
    preview_pair_stats = pairing_metrics["preview"]
    final_pair_stats = pairing_metrics["final"]
    window_match_fraction = float(pairing_metrics["match_fraction"])
    window_overlap_fraction = float(pairing_metrics["overlap_fraction"])
    duplicate_fraction = float(pairing_metrics["duplicate_fraction"])
    count_mismatch = bool(pairing_metrics["count_mismatch"])
    pairing_reason = pairing_metrics["reason"]

    if runtime.pairing_context and window_match_fraction < 0.999999:
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
    if window_overlap_fraction > 0.0 and runtime.pairing_context:
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

    if runtime.pairing_context and runtime.profile_label in {"ci", "release"}:
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
                f"Window count mismatch detected (preview={slices.preview_batches_ct}, final={slices.final_batches_ct})"
            )

    tier = "balanced"
    if config and isinstance(config.context, dict):
        auto_section = config.context.get("auto", {})
        if isinstance(auto_section, dict):
            tier = str(auto_section.get("tier", tier)).lower()

    coverage_summary = assess_bootstrap_coverage_fn(
        tier=tier,
        preview_batches=slices.preview_batches_ct,
        final_batches=slices.final_batches_ct,
        bootstrap_enabled=bool(runtime.bootstrap_enabled),
        bootstrap_replicates=int(runtime.bootstrap_replicates),
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
                "preview_used": slices.preview_batches_ct,
                "preview_required": preview_required,
                "final_used": slices.final_batches_ct,
                "final_required": final_required,
                "replicates_used": runtime.bootstrap_replicates,
                "replicates_required": replicates_required,
            },
        )
        if runtime.pairing_context and runtime.profile_label in {"ci", "release"}:
            raise InvarlockError(
                code="E005",
                message=(
                    "INSUFFICIENT-SAMPLE: bootstrap coverage below policy floors in CI/Release"
                ),
            )

    bootstrap_info = {
        "enabled": bool(runtime.bootstrap_enabled),
        "method": runtime.bootstrap_method,
        "alpha": float(runtime.bootstrap_alpha),
        "replicates": int(runtime.bootstrap_replicates),
        "seed": int(runtime.bootstrap_seed),
        "ci_band": float(runtime.ci_band),
        "window_duplicate_fraction": float(duplicate_fraction),
        "window_match_fraction": float(window_match_fraction),
        "coverage": coverage_summary["coverage"],
    }
    return _PairingCoverageResult(
        preview_pair_stats=preview_pair_stats,
        final_pair_stats=final_pair_stats,
        window_overlap_fraction=float(window_overlap_fraction),
        window_match_fraction=float(window_match_fraction),
        pairing_reason=pairing_reason,
        bootstrap_info=bootstrap_info,
    )


def _pairing_error_result(runtime: Any) -> _PairingCoverageResult:
    return _PairingCoverageResult(
        preview_pair_stats={"matched": 0, "expected": 0},
        final_pair_stats={"matched": 0, "expected": 0},
        window_overlap_fraction=0.0,
        window_match_fraction=1.0,
        pairing_reason=None,
        bootstrap_info={
            "enabled": bool(runtime.bootstrap_enabled),
            "method": runtime.bootstrap_method,
            "alpha": float(runtime.bootstrap_alpha),
            "replicates": int(runtime.bootstrap_replicates),
            "seed": int(runtime.bootstrap_seed),
            "ci_band": float(runtime.ci_band),
        },
    )


__all__ = [
    "_BootstrapDeltaResult",
    "_PairingCoverageResult",
    "_compute_bootstrap_delta_stats",
    "_evaluate_pairing_and_coverage",
    "_pairing_error_result",
]
