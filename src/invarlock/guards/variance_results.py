from __future__ import annotations

import copy
import math
from typing import Any


def _finite_non_bool_values(values: dict[str, float]) -> list[float]:
    return [
        float(value)
        for value in values.values()
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]


def build_scale_statistics(scales: dict[str, float]) -> dict[str, float]:
    values = _finite_non_bool_values(scales)
    if not values:
        return {"mean_scale": 1.0, "min_scale": 1.0, "max_scale": 1.0}
    return {
        "mean_scale": float(sum(values) / len(values)),
        "min_scale": float(min(values)),
        "max_scale": float(max(values)),
    }


def build_prepare_result(
    *,
    policy: dict[str, Any],
    target_modules: dict[str, Any],
    scales: dict[str, float],
    calibration_stats: dict[str, Any],
    preparation_time: float,
    ready: bool,
    warning: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "baseline_metrics": {},
        "policy_applied": policy.copy(),
        "preparation_time": float(preparation_time),
        "ready": bool(ready),
    }
    if ready:
        result["baseline_metrics"] = {
            "target_modules": len(target_modules),
            "proposed_scales": len(scales),
            "scope": policy["scope"],
            "scale_statistics": build_scale_statistics(scales),
            "calibration": calibration_stats.copy(),
        }
    if warning:
        result["warning"] = warning
    if error:
        result["error"] = error
    return result


def evaluate_finalize_state(
    *,
    should_enable: bool,
    enabled_after_ab: bool,
    gate_reason: str,
    ppl_no_ve: float | None,
    ppl_with_ve: float | None,
    final_ppl: float | None,
    ab_windows_used: int | None,
    ab_seed_used: int | None,
    expected_seed: int,
    enable_attempt_count: int,
    disable_attempt_count: int,
    checkpoint_depth: int,
    ab_gain: float | None,
    required_gain_with_deadband: float,
    absolute_floor: float,
    calibration_status: str,
    no_adjustment_required: bool = False,
) -> dict[str, Any]:
    passed = True
    warnings: list[str] = []
    errors: list[str] = []

    if enabled_after_ab != should_enable:
        if should_enable:
            warnings.append(f"VE disabled despite A/B gate approval: {gate_reason}")
        else:
            errors.append(f"VE enabled despite A/B gate rejection: {gate_reason}")
            passed = False

    if not enabled_after_ab and ppl_no_ve and ppl_with_ve and final_ppl is not None:
        ppl_rise = final_ppl - ppl_no_ve
        if ppl_rise > 0.5:
            errors.append(f"Primary-metric rise {ppl_rise:.3f} > 0.5 when VE disabled")
            passed = False

    if (
        ab_windows_used is not None
        and ab_seed_used is not None
        and ab_seed_used != expected_seed
    ):
        warnings.append(
            f"A/B test used unexpected seed {ab_seed_used}, expected {expected_seed}"
        )

    if enable_attempt_count > 3:
        warnings.append(
            f"Multiple enable attempts ({enable_attempt_count}), may indicate instability"
        )
    if disable_attempt_count > 3:
        warnings.append(
            f"Multiple disable attempts ({disable_attempt_count}), may indicate instability"
        )
    if checkpoint_depth > 0:
        warnings.append(f"Uncommitted checkpoints remaining: {checkpoint_depth}")

    if ab_gain is not None and ab_gain > 0 and enabled_after_ab:
        if ab_gain < required_gain_with_deadband:
            errors.append(
                "VE enabled without meeting tie-breaker deadband: "
                f"gain {ab_gain:.3f} < {required_gain_with_deadband:.3f}"
            )
            passed = False

    if ppl_no_ve and ppl_with_ve and enabled_after_ab:
        absolute_improvement = ppl_no_ve - ppl_with_ve
        if absolute_improvement < absolute_floor:
            errors.append(
                "VE enabled without meeting absolute floor: "
                f"improvement {absolute_improvement:.3f} < {absolute_floor}"
            )
            passed = False

    adequate_no_adjustment = bool(
        no_adjustment_required and calibration_status == "no_scaling_required"
    )
    if calibration_status != "complete" and not adequate_no_adjustment:
        warnings.append(
            "Variance calibration coverage insufficient; operating in monitor mode"
        )

    return {"passed": passed, "warnings": warnings, "errors": errors}


def build_finalize_metrics(
    *,
    scales: dict[str, float],
    target_modules: dict[str, Any],
    stats: dict[str, Any],
    focus_modules: set[str],
    enabled_after_ab: bool,
    should_enable: bool,
    ab_gain: float,
    ab_windows_used: int | None,
    ab_seed_used: int | None,
    monitor_only: bool,
    policy: dict[str, Any],
    ppl_no_ve: float | None,
    ppl_with_ve: float | None,
    ratio_ci: tuple[float, float] | None,
    calibration_stats: dict[str, Any],
    predictive_gate_state: dict[str, Any],
    raw_scales_pre_edit: dict[str, float],
    raw_scales_post_edit: dict[str, float],
) -> dict[str, Any]:
    return {
        "proposed_scales": len(scales),
        "target_modules": len(target_modules),
        "target_module_names": stats.get("target_module_names", []),
        "focus_modules": sorted(focus_modules) if focus_modules else [],
        "tap": stats.get("tap"),
        "ve_enabled": enabled_after_ab,
        "ab_gain": ab_gain,
        "ab_windows_used": ab_windows_used,
        "ab_seed_used": ab_seed_used,
        "monitor_only": monitor_only,
        "min_gain_threshold": policy["min_gain"],
        "met_threshold": should_enable,
        "ppl_no_ve": ppl_no_ve,
        "ppl_with_ve": ppl_with_ve,
        "scope": policy["scope"],
        "max_calib_used": policy["max_calib"],
        "mode": policy.get("mode"),
        "min_rel_gain": policy.get("min_rel_gain"),
        "alpha": policy.get("alpha"),
        "ratio_ci": ratio_ci,
        "calibration": calibration_stats.copy(),
        "predictive_gate": predictive_gate_state.copy(),
        "ab_provenance": copy.deepcopy(stats.get("ab_provenance", {})),
        "ab_point_estimates": copy.deepcopy(stats.get("ab_point_estimates", {})),
        "ab_measurements": copy.deepcopy(stats.get("ab_measurements", {})),
        "raw_scales_pre_edit": copy.deepcopy(raw_scales_pre_edit),
        "raw_scales_post_edit": copy.deepcopy(raw_scales_post_edit),
        "proposed_scales_pre_edit": stats.get("proposed_scales_pre_edit", {}),
        "proposed_scales_post_edit": stats.get("proposed_scales_post_edit", {}),
    }


def build_finalize_result(
    *,
    passed: bool,
    metrics: dict[str, Any],
    warnings: list[str],
    errors: list[str],
    finalize_time: float,
    enabled_after_ab: bool,
    ppl_no_ve: float | None,
    scales: dict[str, float],
    stats: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "passed": passed,
        "metrics": metrics,
        "warnings": warnings,
        "errors": errors,
        "finalize_time": float(finalize_time),
        "details": {
            "guard_type": "variance",
            "ve_applied": enabled_after_ab,
            "ab_test_performed": ppl_no_ve is not None,
            "proposed_scales": copy.deepcopy(
                metrics.get("proposed_scales_post_edit", scales)
            ),
            "stats": stats,
            "policy": policy,
        },
    }


__all__ = [
    "build_finalize_metrics",
    "build_finalize_result",
    "build_prepare_result",
    "build_scale_statistics",
    "evaluate_finalize_state",
]
