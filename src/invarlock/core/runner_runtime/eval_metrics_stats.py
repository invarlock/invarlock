from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..bootstrap import (
    INDEPENDENT_SLICE_BOOTSTRAP_METHOD,
    INDEPENDENT_SLICE_BOOTSTRAP_SEED_OFFSET,
    compute_logloss_ci,
)
from ..exceptions import InvarlockError
from ..types import LogLevel
from .pairing import assess_bootstrap_coverage, compute_window_pairing_metrics

_PROFILE_COVERAGE_FLOORS = {
    "ci": {"preview": 0, "final": 0, "replicates": 1200},
    "release": {"preview": 200, "final": 200, "replicates": 3200},
}


@dataclass(frozen=True)
class _BootstrapDeltaResult:
    preview_log_ci: tuple[float, float]
    final_log_ci: tuple[float, float]
    delta_log_ci: tuple[float, float]
    ratio_ci: tuple[float, float]
    delta_ci_method: str
    delta_ci_reason: str | None
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


def _coverage_requirements_with_profile_floor(
    requirements: dict[str, dict[str, int]],
    *,
    tier: str,
    profile: str,
) -> dict[str, dict[str, int]]:
    effective = {
        name: dict(values)
        for name, values in requirements.items()
        if isinstance(name, str) and isinstance(values, dict)
    }
    selected = dict(effective.get(tier) or effective.get("balanced") or {})
    profile_floor = _PROFILE_COVERAGE_FLOORS.get(profile, {})
    for key in ("preview", "final", "replicates"):
        selected[key] = max(
            int(selected.get(key, 0)),
            int(profile_floor.get(key, 0)),
        )
    effective[tier] = selected
    return effective


def _compute_bootstrap_delta_stats(
    runner: Any,
    runtime: Any,
    slices: Any,
    *,
    compute_independent_delta_log_ci_fn: Any,
    logspace_to_ratio_ci_fn: Any,
) -> _BootstrapDeltaResult:
    preview_log_ci = (slices.preview_mean_log, slices.preview_mean_log)
    final_log_ci = (slices.final_mean_log, slices.final_mean_log)
    delta_log_ci = (slices.delta_mean_log, slices.delta_mean_log)
    ratio_ci = (slices.ppl_ratio, slices.ppl_ratio)
    delta_ci_method = "none"
    delta_ci_reason: str | None = None
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

    if (
        runtime.bootstrap_enabled
        and len(slices.final_log_losses) > 0
        and len(slices.preview_log_losses) > 0
    ):
        preview_weights: list[float] | None = None
        final_weights: list[float] | None = None
        if len(slices.preview_token_counts) == len(slices.preview_log_losses):
            preview_weights = [
                float(max(weight, 0)) for weight in slices.preview_token_counts
            ]
        elif slices.preview_token_counts:
            pm_invalid = True
            runner._log_event(
                "eval",
                "preview_slice_weight_mismatch",
                LogLevel.ERROR,
                {
                    "losses": len(slices.preview_log_losses),
                    "weights": len(slices.preview_token_counts),
                },
            )
        if len(slices.final_token_counts) == len(slices.final_log_losses):
            final_weights = [
                float(max(weight, 0)) for weight in slices.final_token_counts
            ]
        elif slices.final_token_counts:
            pm_invalid = True
            runner._log_event(
                "eval",
                "final_slice_weight_mismatch",
                LogLevel.ERROR,
                {
                    "losses": len(slices.final_log_losses),
                    "weights": len(slices.final_token_counts),
                },
            )
        try:
            delta_log_ci = compute_independent_delta_log_ci_fn(
                slices.final_log_losses,
                slices.preview_log_losses,
                final_weights=final_weights,
                preview_weights=preview_weights,
                method="percentile",
                replicates=runtime.bootstrap_replicates,
                alpha=runtime.bootstrap_alpha,
                seed=(runtime.bootstrap_seed + INDEPENDENT_SLICE_BOOTSTRAP_SEED_OFFSET),
            )
            delta_ci_method = INDEPENDENT_SLICE_BOOTSTRAP_METHOD
        except ValueError as exc:
            pm_invalid = True
            delta_ci_reason = "independent_slice_bootstrap_error"
            runner._log_event(
                "eval",
                "independent_slice_delta_error",
                LogLevel.ERROR,
                {"reason": str(exc)},
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

    if len(slices.final_log_losses) == 0 or len(slices.preview_log_losses) == 0:
        degenerate_delta = True
        degenerate_reason = "missing_slice_losses"
        delta_ci_reason = delta_ci_reason or "missing_slice_losses"
        pm_invalid = True
    elif not runtime.bootstrap_enabled:
        delta_ci_reason = "bootstrap_disabled"
    elif math.isclose(
        float(delta_log_ci[0]),
        float(delta_log_ci[1]),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        degenerate_delta = True
        degenerate_reason = "constant_bootstrap_distribution"
    if degenerate_delta:
        runner._log_event(
            "eval",
            "independent_slice_delta_degenerate",
            LogLevel.WARNING,
            {
                "reason": degenerate_reason,
                "preview_windows": len(slices.preview_log_losses),
                "final_windows": len(slices.final_log_losses),
            },
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
    elif pm_invalid:
        degraded_reason = "primary_metric_invalid"
    if needs_pm_fallback or needs_delta_fallback:
        pm_invalid = True

    return _BootstrapDeltaResult(
        preview_log_ci=(float(preview_log_ci[0]), float(preview_log_ci[1])),
        final_log_ci=(float(final_log_ci[0]), float(final_log_ci[1])),
        delta_log_ci=(float(delta_log_ci[0]), float(delta_log_ci[1])),
        ratio_ci=(float(ratio_ci[0]), float(ratio_ci[1])),
        delta_ci_method=delta_ci_method,
        delta_ci_reason=delta_ci_reason,
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

    effective_coverage_requirements = _coverage_requirements_with_profile_floor(
        coverage_requirements,
        tier=tier,
        profile=str(runtime.profile_label or "").strip().lower(),
    )
    coverage_summary = assess_bootstrap_coverage_fn(
        tier=tier,
        preview_batches=slices.preview_batches_ct,
        final_batches=slices.final_batches_ct,
        bootstrap_enabled=bool(runtime.bootstrap_enabled),
        bootstrap_replicates=int(runtime.bootstrap_replicates),
        requirements=effective_coverage_requirements,
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
        "preview_final_delta_basis": "independent_disjoint_slices",
        "preview_final_delta_method": (
            INDEPENDENT_SLICE_BOOTSTRAP_METHOD if runtime.bootstrap_enabled else "none"
        ),
        "preview_final_delta_seed": (
            int(runtime.bootstrap_seed) + INDEPENDENT_SLICE_BOOTSTRAP_SEED_OFFSET
            if runtime.bootstrap_enabled
            else None
        ),
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
            "preview_final_delta_basis": "independent_disjoint_slices",
            "preview_final_delta_method": (
                INDEPENDENT_SLICE_BOOTSTRAP_METHOD
                if runtime.bootstrap_enabled
                else "none"
            ),
            "preview_final_delta_seed": (
                int(runtime.bootstrap_seed) + INDEPENDENT_SLICE_BOOTSTRAP_SEED_OFFSET
                if runtime.bootstrap_enabled
                else None
            ),
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
