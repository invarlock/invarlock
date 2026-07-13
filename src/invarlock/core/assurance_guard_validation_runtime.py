"""Invariant, variance, and guard-metric-impact assurance reconciliation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from invarlock.eval.guard_metric_impact import (
    arm_facts_match_measurements,
    compute_guard_metric_impact,
    degradation_within_limit,
    guard_metric_impact_payload_errors,
)

from .assurance_guard_validation_common import (
    _PASS_STATUSES,
    _finite_number,
    _finite_pair,
    _mapping,
    _nonnegative_int,
    _normalized_token,
    _validate_diagnostics,
)


def _invariants_errors(
    report: Mapping[str, Any], *, require_complete: bool
) -> list[str]:
    invariants = _mapping(report.get("invariants"))
    if invariants is None:
        return []
    errors: list[str] = []
    for field in ("pre", "post"):
        value = invariants.get(field)
        if field not in invariants:
            if require_complete:
                errors.append(f"invariants.{field} is required for strict assurance.")
        elif (
            not isinstance(value, str) or _normalized_token(value) not in _PASS_STATUSES
        ):
            errors.append(f"invariants.{field} must be a passing status.")

    failures = invariants.get("failures")
    if "failures" not in invariants:
        if require_complete:
            errors.append("invariants.failures is required for strict assurance.")
    elif not isinstance(failures, list):
        errors.append("invariants.failures must be an array.")
    elif failures:
        errors.append("invariants.failures must be empty for strict assurance.")

    summary = _mapping(invariants.get("summary"))
    if summary is None:
        if require_complete:
            errors.append("invariants.summary is required for strict assurance.")
        return errors
    for field in ("violations_found", "fatal_violations", "warning_violations"):
        if field not in summary:
            if require_complete:
                errors.append(
                    f"invariants.summary.{field} is required for strict assurance."
                )
            continue
        count = _nonnegative_int(summary.get(field))
        if count is None:
            errors.append(f"invariants.summary.{field} must be an integer.")
        elif count != 0:
            errors.append(f"invariants.summary.{field} must be zero.")
    checks = summary.get("checks_performed")
    if "checks_performed" not in summary:
        if require_complete:
            errors.append(
                "invariants.summary.checks_performed is required for strict assurance."
            )
    else:
        count = _nonnegative_int(checks)
        if count is None:
            errors.append("invariants.summary.checks_performed must be an integer.")
        elif require_complete and count == 0:
            errors.append(
                "invariants.summary.checks_performed must be positive for strict assurance."
            )
    return errors


def _variance_errors(report: Mapping[str, Any], *, require_complete: bool) -> list[str]:
    variance = _mapping(report.get("variance"))
    if variance is None:
        return []
    errors: list[str] = []
    enabled = variance.get("enabled")
    if "enabled" not in variance:
        if require_complete:
            errors.append("variance.enabled is required for strict assurance.")
    elif not isinstance(enabled, bool):
        errors.append("variance.enabled must be a boolean.")

    monitor_only = variance.get("monitor_only")
    if "monitor_only" in variance:
        if not isinstance(monitor_only, bool):
            errors.append("variance.monitor_only must be a boolean.")
        elif monitor_only:
            errors.append("variance.monitor_only cannot pass strict assurance.")

    predictive = _mapping(variance.get("predictive_gate"))
    if predictive is None:
        if require_complete:
            errors.append("variance.predictive_gate is required for strict assurance.")
        return errors
    evaluated = predictive.get("evaluated")
    passed = predictive.get("passed")
    if require_complete or "evaluated" in predictive:
        if not isinstance(evaluated, bool):
            errors.append("variance.predictive_gate.evaluated must be a boolean.")
        elif require_complete and evaluated is not True:
            errors.append(
                "variance.predictive_gate.evaluated must be true for strict assurance."
            )
    if require_complete or "passed" in predictive:
        if not isinstance(passed, bool):
            errors.append("variance.predictive_gate.passed must be a boolean.")
        elif passed is not True:
            errors.append("variance.predictive_gate.passed is false.")

    reason = predictive.get("reason")
    edit = _mapping(report.get("edit"))
    edit_is_noop = bool(
        edit is not None and _normalized_token(str(edit.get("name") or "")) == "noop"
    )
    reason_is_no_adjustment = bool(
        isinstance(reason, str)
        and _normalized_token(reason) == "no-adjustment-required"
    )
    reason_is_gain = bool(
        isinstance(reason, str) and _normalized_token(reason) == "ci-gain-met"
    )
    if passed is True and not (reason_is_no_adjustment or reason_is_gain):
        displayed_reason = reason if isinstance(reason, str) and reason else "missing"
        errors.append(
            f"variance predictive_gate.reason={displayed_reason} cannot be a "
            "passing result."
        )
    if require_complete and edit_is_noop and not reason_is_no_adjustment:
        errors.append(
            "strict no-op variance requires "
            "predictive_gate.reason=no_adjustment_required."
        )
    if require_complete and (reason_is_no_adjustment or edit_is_noop):
        if edit is None or _normalized_token(str(edit.get("name") or "")) != "noop":
            errors.append("variance no_adjustment_required requires edit.name=noop.")

        structure = _mapping(report.get("structure"))
        params_changed = (
            _nonnegative_int(structure.get("params_changed"))
            if structure is not None
            else None
        )
        if params_changed != 0:
            errors.append(
                "variance no_adjustment_required requires "
                "structure.params_changed=0 as an integer."
            )

        if enabled is not False:
            errors.append(
                "variance no_adjustment_required requires variance.enabled=false."
            )
        if monitor_only is not False:
            errors.append(
                "variance no_adjustment_required requires variance.monitor_only=false."
            )

        calibration = _mapping(variance.get("calibration"))
        if calibration is None:
            errors.append(
                "variance no_adjustment_required requires variance.calibration evidence."
            )
        else:
            if calibration.get("status") != "no_scaling_required":
                errors.append(
                    "variance no_adjustment_required requires "
                    "variance.calibration.status=no_scaling_required."
                )
            coverage = _nonnegative_int(calibration.get("coverage"))
            minimum = _nonnegative_int(calibration.get("min_coverage"))
            if (
                minimum is None
                or minimum <= 0
                or coverage is None
                or coverage < minimum
            ):
                errors.append(
                    "variance no_adjustment_required requires adequate variance "
                    "calibration coverage."
                )
    if require_complete and reason_is_gain:
        errors.extend(_variance_gain_errors(variance, predictive, enabled=enabled))
    return errors


def _variance_gain_errors(
    variance: Mapping[str, Any],
    predictive: Mapping[str, Any],
    *,
    enabled: Any,
) -> list[str]:
    errors: list[str] = []
    if enabled is not False:
        errors.append(
            "variance ci_gain_met requires final variance.enabled=false after exact "
            "subject restoration."
        )

    delta_ci = _finite_pair(predictive.get("delta_ci"))
    gain_ci = _finite_pair(predictive.get("gain_ci"))
    mean_delta = _finite_number(predictive.get("mean_delta"))
    if delta_ci is None:
        errors.append("variance ci_gain_met requires a finite two-value delta_ci.")
    if gain_ci is None:
        errors.append("variance ci_gain_met requires a finite two-value gain_ci.")
    if mean_delta is None:
        errors.append("variance ci_gain_met requires a finite mean_delta.")

    policy = _mapping(variance.get("policy"))
    min_effect = (
        _finite_number(policy.get("min_effect_lognll")) if policy is not None else None
    )
    if min_effect is None or min_effect < 0.0:
        errors.append(
            "variance ci_gain_met requires a non-negative finite "
            "policy.min_effect_lognll."
        )
    if policy is None or not isinstance(policy.get("predictive_one_sided"), bool):
        errors.append(
            "variance ci_gain_met requires boolean policy.predictive_one_sided."
        )

    if delta_ci is not None:
        lower, upper = delta_ci
        if lower > upper:
            errors.append("variance predictive_gate.delta_ci must be ordered.")
        if upper >= 0.0:
            errors.append("variance ci_gain_met requires delta_ci strictly below zero.")
        if min_effect is not None and min_effect >= 0.0 and upper > -min_effect:
            errors.append(
                "variance ci_gain_met delta_ci does not meet policy.min_effect_lognll."
            )
    if mean_delta is not None:
        if mean_delta >= 0.0:
            errors.append("variance ci_gain_met requires a negative mean_delta.")
        if min_effect is not None and min_effect >= 0.0 and mean_delta > -min_effect:
            errors.append(
                "variance ci_gain_met mean_delta does not meet "
                "policy.min_effect_lognll."
            )
    if delta_ci is not None and gain_ci is not None:
        expected_gain = (-delta_ci[1], -delta_ci[0])
        if not all(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
            for observed, expected in zip(gain_ci, expected_gain, strict=True)
        ):
            errors.append(
                "variance predictive_gate.gain_ci must be the exact inverse of delta_ci."
            )

    calibration = _mapping(variance.get("calibration"))
    if calibration is None:
        errors.append("variance ci_gain_met requires variance.calibration evidence.")
    else:
        if _normalized_token(str(calibration.get("status") or "")) != "complete":
            errors.append("variance ci_gain_met requires calibration.status=complete.")
        coverage = _nonnegative_int(calibration.get("coverage"))
        minimum = _nonnegative_int(calibration.get("min_coverage"))
        if minimum is None or minimum <= 0 or coverage is None or coverage < minimum:
            errors.append(
                "variance ci_gain_met requires adequate variance calibration coverage."
            )
    return errors


def _guard_metric_impact_errors(
    report: Mapping[str, Any], *, require_complete: bool
) -> list[str]:
    metric_impact = _mapping(report.get("guard_metric_impact"))
    if metric_impact is None or not metric_impact:
        return (
            ["strict assurance missing guard_metric_impact evidence."]
            if require_complete
            else []
        )
    errors: list[str] = []
    stale_fields = {
        "bare_ppl",
        "guarded_ppl",
        "impact_ratio",
        "impact_percent",
        "impact_threshold",
        "metric_direction",
        "impact_basis",
        "impact_value",
    }
    for field in sorted(stale_fields & set(metric_impact)):
        errors.append(
            f"guard_metric_impact.{field} is not part of the guard metric impact contract."
        )
    for field in ("evaluated", "passed"):
        value = metric_impact.get(field)
        if field not in metric_impact:
            if require_complete:
                errors.append(
                    f"guard_metric_impact.{field} is required for strict assurance."
                )
        elif not isinstance(value, bool):
            errors.append(f"guard_metric_impact.{field} must be a boolean.")
        elif require_complete and value is not True:
            errors.append(
                f"guard_metric_impact.{field} must be true for strict assurance."
            )
        elif field == "passed" and value is False:
            errors.append("guard_metric_impact.passed is false.")

    skipped = metric_impact.get("skipped")
    if "skipped" in metric_impact:
        if not isinstance(skipped, bool):
            errors.append("guard_metric_impact.skipped must be a boolean.")
        elif require_complete and skipped:
            errors.append(
                "strict assurance requires measured guard_metric_impact evidence."
            )
    mode = metric_impact.get("mode")
    if (
        isinstance(mode, str)
        and _normalized_token(mode) == "skipped"
        and require_complete
    ):
        errors.append(
            "strict assurance requires measured guard_metric_impact evidence."
        )

    degradation_limit = _finite_number(metric_impact.get("degradation_limit"))
    if "degradation_limit" not in metric_impact:
        if require_complete:
            errors.append(
                "guard_metric_impact.degradation_limit is required for strict assurance."
            )
    elif degradation_limit is None or degradation_limit < 0.0:
        errors.append(
            "guard_metric_impact.degradation_limit must be finite and non-negative."
        )
        degradation_limit = None

    required_measurement_fields = (
        "metric_kind",
        "direction",
        "bare_value",
        "guarded_value",
        "degradation_basis",
        "degradation",
        "display_value",
        "display_unit",
        "bare_facts",
        "guarded_facts",
    )
    if require_complete:
        for field in required_measurement_fields:
            if field not in metric_impact:
                errors.append(
                    f"guard_metric_impact.{field} is required for strict assurance."
                )
    metric_kind = metric_impact.get("metric_kind")
    direction = metric_impact.get("direction")
    degradation_basis = metric_impact.get("degradation_basis")
    bare_value = metric_impact.get("bare_value")
    guarded_value = metric_impact.get("guarded_value")
    degradation = _finite_number(metric_impact.get("degradation"))
    display_value = _finite_number(metric_impact.get("display_value"))
    display_unit = metric_impact.get("display_unit")
    measurement = compute_guard_metric_impact(
        metric_kind,
        bare_value,
        guarded_value,
    )
    if measurement is None:
        errors.append(
            "guard_metric_impact retained measurements are invalid or unsupported."
        )
    else:
        if not arm_facts_match_measurements(
            metric_kind,
            metric_impact.get("bare_facts"),
            metric_impact.get("guarded_facts"),
            bare_value,
            guarded_value,
        ):
            errors.append(
                "guard_metric_impact arm facts do not replay the paired measurements."
            )
        if direction != measurement.direction:
            errors.append("guard_metric_impact.direction disagrees with metric_kind.")
        if degradation_basis != measurement.degradation_basis:
            errors.append(
                "guard_metric_impact.degradation_basis disagrees with metric_kind."
            )
        if degradation is None:
            errors.append("guard_metric_impact.degradation must be finite.")
        elif not math.isclose(
            degradation,
            measurement.degradation,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            errors.append(
                "guard_metric_impact.degradation disagrees with retained measurements."
            )
        if display_value is None or not math.isclose(
            display_value,
            measurement.display_value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            errors.append(
                "guard_metric_impact.display_value disagrees with retained measurements."
            )
        if display_unit != measurement.display_unit:
            errors.append(
                "guard_metric_impact.display_unit disagrees with metric_kind."
            )
        if (
            degradation is not None
            and degradation_limit is not None
            and not degradation_within_limit(
                degradation=degradation,
                degradation_limit=degradation_limit,
            )
        ):
            errors.append("guard_metric_impact.degradation exceeds degradation_limit.")

    checks = metric_impact.get("checks")
    if checks is not None:
        if not isinstance(checks, dict):
            errors.append("guard_metric_impact.checks must be an object.")
        else:
            for name, passed in checks.items():
                if not isinstance(passed, bool):
                    errors.append(
                        f"guard_metric_impact.checks.{name} must be a boolean."
                    )
                elif passed is False:
                    errors.append(f"guard_metric_impact.checks.{name} is false.")
    errors.extend(
        _validate_diagnostics("guard_metric_impact", metric_impact.get("diagnostics"))
    )
    errors.extend(
        guard_metric_impact_payload_errors(
            metric_impact,
            subject_report=report,
            require_bare_report=require_complete,
        )
    )
    return errors
