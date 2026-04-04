from __future__ import annotations

import math
from typing import Any

from ._contracts import guard_assert


def _is_non_bool_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _coerce_non_bool_float(value: Any) -> float | None:
    return float(value) if _is_non_bool_finite_number(value) else None


def predictive_gate_outcome(
    mean_delta: float,
    delta_ci: tuple[float, float] | None,
    min_effect: float,
    one_sided: bool,
) -> tuple[bool, str]:
    """Decide whether the predictive gate passes given the CI and tier semantics."""
    guard_assert(min_effect >= 0.0, "variance.min_effect must be >= 0")
    if (
        delta_ci is None
        or len(delta_ci) != 2
        or not all(_is_non_bool_finite_number(val) for val in delta_ci)
    ):
        return False, "ci_unavailable"

    lower = float(delta_ci[0])
    upper = float(delta_ci[1])
    min_effect = float(min_effect or 0.0)

    if one_sided:
        if upper >= 0.0:
            return False, "ci_contains_zero"
        if mean_delta >= 0.0:
            return False, "mean_not_negative"
        if upper > -min_effect:
            return False, "gain_below_threshold"
        if mean_delta > -min_effect:
            return False, "gain_below_threshold"
        return True, "ci_gain_met"

    if lower <= 0.0 <= upper:
        return False, "ci_contains_zero"
    if lower > 0.0:
        if lower >= min_effect and mean_delta >= min_effect:
            return False, "regression_detected"
        return False, "mean_not_negative"
    if upper > -min_effect:
        return False, "gain_below_threshold"
    if mean_delta >= 0.0:
        return False, "mean_not_negative"
    if mean_delta > -min_effect:
        return False, "gain_below_threshold"
    return True, "ci_gain_met"


def refresh_calibration_defaults(guard: Any) -> None:
    """Ensure calibration config contains required defaults."""
    default_calibration = {
        "windows": 6,
        "min_coverage": 4,
        "seed": guard._policy.get("seed", 123),
    }
    calibration_cfg = guard._policy.get("calibration", {}) or {}
    if not isinstance(calibration_cfg, dict):
        calibration_cfg = {}
    guard._policy["calibration"] = {**default_calibration, **calibration_cfg}


def set_ab_results(
    guard: Any,
    ppl_no_ve: float,
    ppl_with_ve: float,
    windows_used: int | None = None,
    seed_used: int | None = None,
    ratio_ci: tuple[float, float] | None = None,
) -> None:
    """Store A/B testing results with reinforced validation logic."""
    guard._ppl_no_ve = ppl_no_ve
    guard._ppl_with_ve = ppl_with_ve
    guard._ab_windows_used = windows_used
    guard._ab_seed_used = seed_used
    guard._ratio_ci = ratio_ci

    if ppl_no_ve is None or ppl_with_ve is None or ppl_no_ve <= 0:
        guard._ab_gain = 0.0
        gain_status = "invalid_ppl"
    else:
        try:
            guard._ab_gain = (ppl_no_ve - ppl_with_ve) / max(ppl_no_ve, 1e-9)
            if not (
                isinstance(guard._ab_gain, int | float)
                and abs(guard._ab_gain) < float("inf")
            ):
                guard._ab_gain = 0.0
                gain_status = "numeric_error"
            else:
                gain_status = "computed"
        except (ZeroDivisionError, OverflowError, TypeError):
            guard._ab_gain = 0.0
            gain_status = "numeric_error"

    ppl_no_ve_str = f"{ppl_no_ve:.3f}" if ppl_no_ve is not None else "None"
    ppl_with_ve_str = f"{ppl_with_ve:.3f}" if ppl_with_ve is not None else "None"

    guard._log_event(
        "ab_results_stored",
        message=(
            f"A/B results: {ppl_no_ve_str} → {ppl_with_ve_str} "
            f"(gain: {guard._ab_gain:.3f}, status: {gain_status})"
        ),
        ppl_no_ve=ppl_no_ve,
        ppl_with_ve=ppl_with_ve,
        gain=guard._ab_gain,
        gain_status=gain_status,
        windows_used=windows_used,
        seed_used=seed_used,
        ratio_ci=ratio_ci,
    )
    guard._post_edit_evaluated = True

    upper_ratio = None
    if isinstance(ratio_ci, tuple | list) and len(ratio_ci) == 2:
        upper_ratio = _coerce_non_bool_float(ratio_ci[1])

    if upper_ratio is not None and upper_ratio < 1.0:
        guard._predictive_gate_state.update(
            {"evaluated": True, "passed": True, "reason": "manual_override"}
        )


def evaluate_ab_gate(guard: Any) -> tuple[bool, str]:
    """Evaluate the A/B gate decision with reinforced criteria."""
    mode = guard._policy.get("mode", "ci")
    min_rel_gain = guard._policy.get("min_rel_gain", 0.0)
    tie_breaker = float(
        guard._policy.get("tie_breaker_deadband", guard.TIE_BREAKER_DEADBAND)
    )
    min_effect_log = guard._policy.get("min_effect_lognll")

    predictive_enabled = bool(guard._policy.get("predictive_gate", True))
    gate_state = getattr(guard, "_predictive_gate_state", {}) or {}
    if (
        predictive_enabled
        and not gate_state.get("evaluated")
        and guard._ratio_ci is not None
    ):
        gate_state = {
            **gate_state,
            "evaluated": True,
            "passed": True,
            "reason": gate_state.get("reason", "synthetic_ab_gate"),
        }
        guard._predictive_gate_state = gate_state

    if guard._ab_gain is None:
        return False, "no_ab_results"

    if (
        guard._ppl_no_ve is None
        or guard._ppl_with_ve is None
        or guard._ppl_no_ve <= 0
        or guard._ppl_with_ve <= 0
    ):
        return False, "invalid_ppl_values"

    relative_gain = guard._ab_gain
    if relative_gain < min_rel_gain:
        return (
            False,
            f"below_min_rel_gain (gain={relative_gain:.3f} < {min_rel_gain:.3f})",
        )

    if min_effect_log is not None:
        log_gain = math.log(max(guard._ppl_no_ve, 1e-9)) - math.log(
            max(guard._ppl_with_ve, 1e-9)
        )
        if log_gain < float(min_effect_log):
            return (
                False,
                f"below_min_effect_lognll (gain={log_gain:.6f} < {float(min_effect_log):.6f})",
            )

    if mode == "ci":
        if guard._ratio_ci is None:
            return False, "missing_ratio_ci"
        ratio_lo, ratio_hi = guard._ratio_ci
        if not all(
            _is_non_bool_finite_number(value) and value > 0
            for value in (ratio_lo, ratio_hi)
        ):
            return False, "invalid_ratio_ci"
        required_hi = 1.0 - min_rel_gain
        if min_effect_log is not None:
            required_hi = min(required_hi, math.exp(-float(min_effect_log)))
        if ratio_hi > required_hi:
            return (
                False,
                f"ci_interval_too_high (hi={ratio_hi:.3f} > {required_hi:.3f})",
            )

    absolute_improvement = guard._ppl_no_ve - guard._ppl_with_ve
    if absolute_improvement < guard.ABSOLUTE_FLOOR:
        return (
            False,
            f"below_absolute_floor (improvement={absolute_improvement:.3f} < {guard.ABSOLUTE_FLOOR})",
        )

    required_gain = guard._policy["min_gain"] + tie_breaker
    if guard._ab_gain < required_gain:
        return (
            False,
            f"below_threshold_with_deadband (gain={guard._ab_gain:.3f} < {required_gain:.3f})",
        )

    if predictive_enabled and not gate_state.get("passed", False):
        reason = gate_state.get("reason", "predictive_gate_failed")
        return False, f"predictive_gate_failed ({reason})"

    return (
        True,
        f"criteria_met (gain={guard._ab_gain:.3f} >= {required_gain:.3f}, improvement={absolute_improvement:.3f})",
    )


__all__ = [
    "evaluate_ab_gate",
    "predictive_gate_outcome",
    "refresh_calibration_defaults",
    "set_ab_results",
]
