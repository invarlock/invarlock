from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from invarlock.core.bootstrap import compute_paired_delta_log_ci

from .variance_batching import safe_mean
from .variance_policy import predictive_gate_outcome

_VARIANCE_EVALUATION_ERRORS = (
    ArithmeticError,
    AttributeError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def evaluate_calibration_pass(
    guard: Any,
    model: nn.Module,
    calibration_batches: list[Any],
    min_coverage: int,
    calib_seed: int,
    tag: str,
    *,
    compute_paired_delta_log_ci_fn: Any = compute_paired_delta_log_ci,
) -> None:
    """Run deterministic calibration for A/B evaluation and predictive gating."""
    predictive_state: dict[str, Any] = {
        "evaluated": False,
        "passed": not bool(guard._policy.get("predictive_gate", True)),
        "reason": "disabled"
        if not bool(guard._policy.get("predictive_gate", True))
        else "no_calibration",
        "delta_ci": (None, None),
        "gain_ci": (None, None),
        "mean_delta": None,
    }

    requested = len(calibration_batches)
    guard._calibration_stats.update(
        {
            "requested": requested,
            "coverage": 0,
            "min_coverage": min_coverage,
            "seed": calib_seed,
            "status": "no_calibration" if not calibration_batches else "insufficient",
            "tag": tag,
        }
    )
    guard._stats.setdefault("calibration", {})
    guard._stats["calibration"].update(
        {
            "requested": requested,
            "min_coverage": min_coverage,
            "seed": calib_seed,
            "tag": tag,
        }
    )

    fingerprint = guard._fingerprint_targets()
    if fingerprint:
        guard._stats["target_fingerprint"] = fingerprint

    if not calibration_batches:
        guard._ratio_ci = None
        guard._predictive_gate_state = predictive_state
        guard._stats["predictive_gate"] = predictive_state.copy()
        return

    device = next(model.parameters()).device
    torch.manual_seed(calib_seed)
    ppl_no_ve_samples, loss_no_ve_samples, token_counts = (
        guard._compute_ppl_for_batches(
            model, calibration_batches, device, return_counts=True
        )
    )
    coverage = min(len(calibration_batches), len(ppl_no_ve_samples))
    ppl_with_ve_samples: list[float] = []
    loss_with_ve_samples: list[float] = []
    token_counts_with: list[int] = []

    enable_success = False
    if coverage >= min_coverage and guard._scales:
        prev_enable_attempts = guard._enable_attempt_count
        prev_disable_attempts = guard._disable_attempt_count
        prev_prepared_flag = guard._prepared
        try:
            guard._prepared = True
            enable_success = guard.enable(model)
        finally:
            guard._prepared = prev_prepared_flag
        try:
            torch.manual_seed(calib_seed)
            if enable_success:
                ppl_with_ve_samples, loss_with_ve_samples, token_counts_with = (
                    guard._compute_ppl_for_batches(
                        model, calibration_batches, device, return_counts=True
                    )
                )
        finally:
            if enable_success:
                guard.disable(model)
        guard._enable_attempt_count = prev_enable_attempts
        guard._disable_attempt_count = prev_disable_attempts

    coverage = min(
        coverage,
        len(ppl_with_ve_samples) if ppl_with_ve_samples else coverage,
        len(loss_with_ve_samples) if loss_with_ve_samples else coverage,
        len(token_counts) if token_counts else coverage,
        len(token_counts_with) if token_counts_with else coverage,
    )
    guard._calibration_stats.update(
        {
            "coverage": coverage,
            "status": "insufficient" if coverage < min_coverage else "pending",
        }
    )

    window_ids = guard._calibration_window_ids
    status_a = "evaluated" if coverage > 0 else "no_data"
    guard._record_ab_provenance(
        "condition_a",
        tag=tag,
        mode="edited_no_ve",
        window_ids=window_ids,
        fingerprint=fingerprint,
        status=status_a,
    )

    if coverage >= min_coverage and not guard._scales:
        ppl_no_ve_samples = ppl_no_ve_samples[:coverage]
        ppl_no_ve_mean = safe_mean(ppl_no_ve_samples)
        if ppl_no_ve_mean is None:
            guard._ratio_ci = None
            predictive_state["reason"] = "no_valid_samples"
            guard._predictive_gate_state = predictive_state
            guard._stats["predictive_gate"] = predictive_state.copy()
            return
        guard.set_ab_results(
            ppl_no_ve=ppl_no_ve_mean,
            ppl_with_ve=ppl_no_ve_mean,
            windows_used=coverage,
            seed_used=calib_seed,
            ratio_ci=(1.0, 1.0),
        )
        guard._calibration_stats.update(
            {
                "status": "no_scaling_required",
                "ppl_no_ve": ppl_no_ve_mean,
                "ratio_ci": (1.0, 1.0),
            }
        )
        guard._stats["ab_point_estimates"] = {
            "tag": tag,
            "ppl_no_ve": ppl_no_ve_mean,
            "ppl_with_ve": ppl_no_ve_mean,
        }
        guard._record_ab_provenance(
            "condition_b",
            tag=tag,
            mode="virtual_ve",
            window_ids=window_ids,
            fingerprint=fingerprint,
            status="no_scales",
        )
        predictive_state.update(
            {"evaluated": True, "passed": False, "reason": "no_scales"}
        )
        guard._predictive_gate_state = predictive_state
        guard._stats["predictive_gate"] = predictive_state.copy()
        return

    if coverage >= min_coverage and ppl_with_ve_samples and loss_with_ve_samples:
        ppl_no_ve_samples = ppl_no_ve_samples[:coverage]
        loss_no_ve_samples = loss_no_ve_samples[:coverage]
        ppl_with_ve_samples = ppl_with_ve_samples[:coverage]
        loss_with_ve_samples = loss_with_ve_samples[:coverage]
        token_counts = token_counts[:coverage]

        ratios = [
            with_val / no_val
            for with_val, no_val in zip(
                ppl_with_ve_samples, ppl_no_ve_samples, strict=False
            )
            if no_val > 0
        ]
        if ratios:
            ratio_ci = guard._bootstrap_mean_ci(
                ratios,
                alpha=guard._policy.get("alpha", 0.05),
                n_bootstrap=500,
                seed=calib_seed,
            )
            ppl_no_ve_mean = safe_mean(ppl_no_ve_samples) or 0.0
            ppl_with_ve_mean = safe_mean(ppl_with_ve_samples) or 0.0
            guard.set_ab_results(
                ppl_no_ve=ppl_no_ve_mean,
                ppl_with_ve=ppl_with_ve_mean,
                windows_used=coverage,
                seed_used=calib_seed,
                ratio_ci=ratio_ci,
            )
            guard._calibration_stats.update(
                {
                    "status": "complete",
                    "ppl_no_ve": ppl_no_ve_mean,
                    "ppl_with_ve": ppl_with_ve_mean,
                    "ratio_ci": ratio_ci,
                }
            )
            guard._record_ab_provenance(
                "condition_b",
                tag=tag,
                mode="virtual_ve",
                window_ids=window_ids,
                fingerprint=fingerprint,
                status="evaluated",
            )
            guard._stats["ab_point_estimates"] = {
                "tag": tag,
                "ppl_no_ve": ppl_no_ve_mean,
                "ppl_with_ve": ppl_with_ve_mean,
                "coverage": coverage,
            }

        delta_ci: tuple[float, float] | None = None
        try:
            delta_ci = compute_paired_delta_log_ci_fn(
                loss_with_ve_samples,
                loss_no_ve_samples,
                weights=token_counts,
                method="bca",
                replicates=500,
                alpha=guard._policy.get("alpha", 0.05),
                seed=calib_seed + 211,
            )
        except _VARIANCE_EVALUATION_ERRORS as exc:
            delta_ci = None
            guard._log_event(
                "predictive_gate_error",
                level="WARN",
                message="Failed to compute predictive ΔlogNLL CI",
                error=str(exc),
            )

        predictive_state["evaluated"] = True
        if token_counts:
            sw = 0.0
            swx = 0.0
            for with_loss, no_loss, weight in zip(
                loss_with_ve_samples, loss_no_ve_samples, token_counts, strict=False
            ):
                sw += float(weight)
                swx += float(weight) * (with_loss - no_loss)
            mean_delta = float(swx / sw) if sw > 0 else float("nan")
        else:
            mean_delta = float(
                np.mean(
                    [
                        with_loss - no_loss
                        for with_loss, no_loss in zip(
                            loss_with_ve_samples, loss_no_ve_samples, strict=False
                        )
                    ]
                )
            )
        predictive_state["mean_delta"] = mean_delta

        if delta_ci is not None and all(
            isinstance(val, (int | float)) and math.isfinite(val) for val in delta_ci
        ):
            delta_ci = (float(delta_ci[0]), float(delta_ci[1]))
            predictive_state["delta_ci"] = delta_ci
            predictive_state["gain_ci"] = (-delta_ci[1], -delta_ci[0])
            if not guard._policy.get("predictive_gate", True):
                predictive_state["passed"] = True
                predictive_state["reason"] = "disabled"
            else:
                one_sided = bool(guard._policy.get("predictive_one_sided", False))
                min_effect = float(guard._policy.get("min_effect_lognll", 0.0) or 0.0)
                passed, reason = predictive_gate_outcome(
                    mean_delta=mean_delta,
                    delta_ci=delta_ci,
                    min_effect=min_effect,
                    one_sided=one_sided,
                )
                predictive_state["passed"] = passed
                predictive_state["reason"] = reason
        else:
            predictive_state["delta_ci"] = (None, None)
            predictive_state["gain_ci"] = (None, None)
            predictive_state["reason"] = (
                predictive_state.get("reason", "ci_unavailable")
                if predictive_state.get("reason") != "disabled"
                else "disabled"
            )
            if guard._calibration_stats.get("status") == "complete":
                guard._calibration_stats["status"] = "pending"
    else:
        guard._ratio_ci = None
        guard._log_event(
            "prepare_monitor_mode",
            level="WARN",
            message="VE calibration coverage insufficient; guard will monitor only",
            requested=requested,
            coverage=coverage,
            min_coverage=min_coverage,
            tag=tag,
        )
        if predictive_state.get("reason") not in {"disabled"}:
            if coverage < min_coverage:
                predictive_state["reason"] = "insufficient_coverage"
            elif not ppl_with_ve_samples:
                predictive_state["reason"] = "ve_enable_failed"

        if "condition_b" not in guard._stats.get("ab_provenance", {}):
            guard._record_ab_provenance(
                "condition_b",
                tag=tag,
                mode="virtual_ve",
                window_ids=window_ids,
                fingerprint=fingerprint,
                status="not_evaluated",
            )

    if (
        "ab_point_estimates" not in guard._stats
        or guard._stats["ab_point_estimates"].get("tag") != tag
    ):
        ppl_no_ve_mean = (
            float(np.mean(ppl_no_ve_samples[:coverage])) if coverage > 0 else None
        )
        ppl_with_ve_mean = (
            float(np.mean(ppl_with_ve_samples[:coverage]))
            if ppl_with_ve_samples and coverage > 0
            else None
        )
        guard._stats["ab_point_estimates"] = {
            "tag": tag,
            "ppl_no_ve": ppl_no_ve_mean,
            "ppl_with_ve": ppl_with_ve_mean,
            "coverage": coverage,
        }

    guard._predictive_gate_state = predictive_state
    guard._stats["predictive_gate"] = predictive_state.copy()


def refresh_after_edit_metrics(
    guard: Any,
    model: nn.Module,
    tag: str = "post_edit",
    adapter: Any | None = None,
) -> None:
    """Ensure VE metrics are recomputed on the edited model."""
    if not guard._prepared:
        return
    if guard._post_edit_evaluated and tag == "post_edit":
        return
    if not guard._calibration_batches:
        guard._log_event(
            "post_edit_calibration_skipped",
            level="WARN",
            message="Skipping post-edit VE evaluation (no calibration batches)",
        )
        guard._post_edit_evaluated = True
        return

    adapter_ref = adapter or guard._adapter_ref
    guard._target_modules = guard._resolve_target_modules(model, adapter_ref)
    guard._stats["target_module_names"] = sorted(guard._target_modules.keys())

    try:
        guard._scales = guard._compute_variance_scales(
            model, guard._calibration_batches
        )
    except _VARIANCE_EVALUATION_ERRORS as exc:
        guard._log_event(
            "post_edit_scale_failure",
            level="ERROR",
            message="Failed to recompute VE scales after edit",
            error=str(exc),
        )
        guard._scales = {}

    if guard._focus_modules:
        guard._scales = {
            name: scale
            for name, scale in guard._scales.items()
            if guard._is_focus_match(name)
        }

    guard._stats.setdefault("target_module_names", sorted(guard._target_modules.keys()))
    guard._stats["target_modules_post_edit"] = list(guard._target_modules.keys())
    normalized_post_scales = {
        guard._normalize_scale_name(name): scale
        for name, scale in guard._scales.items()
    }
    guard._stats["proposed_scales_post_edit"] = normalized_post_scales.copy()
    guard._stats["raw_scales_post_edit"] = guard._raw_scales.copy()
    guard._stats["raw_scales_post_edit_normalized"] = {
        guard._normalize_scale_name(name): scale
        for name, scale in guard._raw_scales.items()
    }
    guard._raw_scales_post_edit = {
        guard._normalize_scale_name(name): scale
        for name, scale in guard._raw_scales.items()
        if guard._is_focus_match(name)
    }
    if normalized_post_scales:
        guard._log_event(
            "post_edit_scales",
            message="Post-edit VE proposed scales",
            count=len(normalized_post_scales),
            min_scale=min(normalized_post_scales.values()),
            max_scale=max(normalized_post_scales.values()),
        )

    calibration_cfg = guard._policy.get("calibration", {})
    requested_windows = int(calibration_cfg.get("windows", 0) or 0)
    min_coverage = int(
        calibration_cfg.get(
            "min_coverage",
            max(1, requested_windows // 2 if requested_windows else 1),
        )
    )
    calib_seed = int(calibration_cfg.get("seed", guard._policy.get("seed", 123)))

    guard._calibration_stats = {
        "requested": len(guard._calibration_batches)
        if requested_windows == 0
        else requested_windows,
        "coverage": 0,
        "min_coverage": min_coverage,
        "seed": calib_seed,
        "status": "pending",
        "tag": tag,
    }
    guard._evaluate_calibration_pass(
        model, guard._calibration_batches, min_coverage, calib_seed, tag
    )
    guard._post_edit_evaluated = True


__all__ = ["evaluate_calibration_pass", "refresh_after_edit_metrics"]
