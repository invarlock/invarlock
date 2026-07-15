"""Strict reconciliation of variance summaries with raw guard evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .assurance_guard_validation_common import (
    _finite_number,
    _finite_pair,
    _mapping,
    _nonnegative_int,
    _normalized_token,
)
from .assurance_guard_validation_variance_binding import (
    _variance_policy_semantic_errors,
    _variance_report_binding_errors,
)
from .assurance_guard_validation_variance_measurements import (
    _variance_measurement_errors,
)
from .assurance_guard_validation_variance_noop import (
    _variance_noop_evidence_errors,
)
from .assurance_guard_validation_variance_provenance import (
    _variance_ab_provenance_errors,
    _variance_details_mirror_errors,
    _variance_policy_calibration_errors,
)
from .assurance_guard_validation_variance_scales import (
    _variance_gain_scale_errors,
)


def _variance_inventory_errors(
    report: Mapping[str, Any],
    variance: Mapping[str, Any],
    inventory: list[tuple[str, dict[str, Any], str]],
    *,
    require_complete: bool,
    enforce_outcome: bool = True,
) -> list[str]:
    _ = enforce_outcome
    matches = [
        (entry, source) for name, entry, source in inventory if name == "variance"
    ]
    if len(matches) != 1:
        return (
            ["strict assurance requires exactly one raw variance guard entry."]
            if require_complete
            else []
        )

    entry, source = matches[0]
    metrics = _mapping(entry.get("metrics"))
    if metrics is None:
        return (
            [f"{source}.metrics is required for strict variance assurance."]
            if require_complete
            else []
        )

    errors: list[str] = []
    expected_provenance: dict[str, Any] = {}

    def reconcile(
        summary: Mapping[str, Any],
        raw: Mapping[str, Any],
        key: str,
        *,
        raw_key: str | None = None,
        summary_path: str,
        raw_path: str,
        required: bool = True,
    ) -> None:
        mirror_key = raw_key or key
        if key not in summary or mirror_key not in raw:
            if require_complete and required:
                errors.append(
                    f"{raw_path}.{mirror_key} and {summary_path}.{key} are required "
                    "for strict assurance."
                )
            elif key in summary or mirror_key in raw:
                errors.append(
                    f"{raw_path}.{mirror_key} must match {summary_path}.{key} exactly."
                )
            return
        if key == "ratio_ci":
            summary_pair = _finite_pair(summary.get(key))
            raw_pair = _finite_pair(raw.get(mirror_key))
            if summary_pair is not None and summary_pair == raw_pair:
                return
        if raw.get(mirror_key) != summary.get(key):
            errors.append(
                f"{raw_path}.{mirror_key} must match {summary_path}.{key} exactly."
            )

    reconcile(
        variance,
        metrics,
        "enabled",
        raw_key="ve_enabled",
        summary_path="variance",
        raw_path=f"{source}.metrics",
    )
    reconcile(
        variance,
        metrics,
        "monitor_only",
        summary_path="variance",
        raw_path=f"{source}.metrics",
    )

    predictive = _mapping(variance.get("predictive_gate"))
    raw_predictive = _mapping(metrics.get("predictive_gate"))
    if predictive is None or raw_predictive is None:
        if require_complete:
            errors.append(
                f"{source}.metrics.predictive_gate and variance.predictive_gate are "
                "required for strict assurance."
            )
    else:
        for key in ("evaluated", "passed", "reason"):
            reconcile(
                predictive,
                raw_predictive,
                key,
                summary_path="variance.predictive_gate",
                raw_path=f"{source}.metrics.predictive_gate",
            )
        for key in ("delta_ci", "gain_ci", "mean_delta"):
            reconcile(
                predictive,
                raw_predictive,
                key,
                summary_path="variance.predictive_gate",
                raw_path=f"{source}.metrics.predictive_gate",
                required=False,
            )

    calibration = _mapping(variance.get("calibration"))
    raw_calibration = _mapping(metrics.get("calibration"))
    if calibration is None or raw_calibration is None:
        if require_complete:
            errors.append(
                f"{source}.metrics.calibration and variance.calibration are required "
                "for strict assurance."
            )
    else:
        for key in ("status", "coverage", "min_coverage"):
            reconcile(
                calibration,
                raw_calibration,
                key,
                summary_path="variance.calibration",
                raw_path=f"{source}.metrics.calibration",
            )

    reason = predictive.get("reason") if predictive is not None else None
    if (
        require_complete
        and isinstance(reason, str)
        and _normalized_token(reason) == "no-adjustment-required"
    ):
        errors.extend(
            _variance_noop_evidence_errors(
                report,
                variance,
                entry,
                metrics,
                source=source,
            )
        )
    normalized_reason = _normalized_token(reason) if isinstance(reason, str) else ""
    replayable_predictive_reasons = {
        "ci-gain-met",
        "ci-contains-zero",
        "gain-below-threshold",
        "mean-not-negative",
        "regression-detected",
    }
    if require_complete and normalized_reason in replayable_predictive_reasons:
        for top_key, raw_key in (
            ("ve_enabled_during_validation", "ve_enabled_during_validation"),
            ("subject_restored_after_ab", "subject_restored_after_ab"),
            ("met_threshold", "met_threshold"),
            ("gain", "ab_gain"),
            ("ppl_no_ve", "ppl_no_ve"),
            ("ppl_with_ve", "ppl_with_ve"),
            ("ratio_ci", "ratio_ci"),
        ):
            reconcile(
                variance,
                metrics,
                top_key,
                raw_key=raw_key,
                summary_path="variance",
                raw_path=f"{source}.metrics",
            )
        if calibration is not None and raw_calibration is not None:
            for key in ("requested", "seed"):
                reconcile(
                    calibration,
                    raw_calibration,
                    key,
                    summary_path="variance.calibration",
                    raw_path=f"{source}.metrics.calibration",
                )

        top_policy = _mapping(variance.get("policy"))
        raw_policy = _mapping(entry.get("policy"))
        if top_policy is None or raw_policy is None:
            errors.append(
                f"{source}.policy and variance.policy are required for ci_gain_met."
            )
        elif raw_policy != top_policy:
            errors.append(f"{source}.policy must match variance.policy exactly.")

        details_errors, details_stats = _variance_details_mirror_errors(
            entry,
            metrics,
            source=source,
        )
        errors.extend(details_errors)
        errors.extend(
            _variance_policy_calibration_errors(
                raw_policy,
                metrics,
                source=source,
            )
        )
        errors.extend(
            _variance_policy_semantic_errors(raw_policy, metrics, source=source)
        )
        report_errors, expected_provenance = _variance_report_binding_errors(
            report,
            raw_policy,
            source=source,
        )
        errors.extend(report_errors)

        ab_test = _mapping(variance.get("ab_test"))
        if ab_test is None:
            errors.append("variance.ab_test is required for ci_gain_met assurance.")
        else:
            reconcile(
                ab_test,
                metrics,
                "seed",
                raw_key="ab_seed_used",
                summary_path="variance.ab_test",
                raw_path=f"{source}.metrics",
            )
            reconcile(
                ab_test,
                metrics,
                "windows_used",
                raw_key="ab_windows_used",
                summary_path="variance.ab_test",
                raw_path=f"{source}.metrics",
            )
            top_points = _mapping(ab_test.get("point_estimates"))
            raw_points = _mapping(metrics.get("ab_point_estimates"))
            if top_points is None or raw_points is None or top_points != raw_points:
                errors.append(
                    f"{source}.metrics.ab_point_estimates must match "
                    "variance.ab_test.point_estimates exactly."
                )
            top_provenance = _mapping(ab_test.get("provenance"))
            raw_provenance = _mapping(metrics.get("ab_provenance"))
            if top_provenance is None or raw_provenance is None:
                errors.append(
                    f"{source}.metrics.ab_provenance and variance.ab_test.provenance "
                    "are required for ci_gain_met."
                )
            else:
                for condition in ("condition_a", "condition_b"):
                    if top_provenance.get(condition) != raw_provenance.get(condition):
                        errors.append(
                            f"{source}.metrics.ab_provenance.{condition} must match "
                            f"variance.ab_test.provenance.{condition} exactly."
                        )

        errors.extend(
            _variance_ab_provenance_errors(
                metrics,
                _nonnegative_int(raw_calibration.get("coverage"))
                if raw_calibration is not None
                else None,
                source=source,
                top_provenance=(
                    _mapping(ab_test.get("provenance")) if ab_test is not None else None
                ),
                details_stats=details_stats,
                condition_b_statuses=frozenset({"evaluated"}),
                expected_provenance=expected_provenance,
            )
        )
        errors.extend(
            _variance_measurement_errors(
                variance,
                entry,
                metrics,
                (
                    _nonnegative_int(raw_calibration.get("coverage"))
                    if raw_calibration is not None
                    else None
                ),
                raw_policy,
                source=source,
                no_adjustment=False,
            )
        )
        errors.extend(
            _variance_gain_scale_errors(
                entry,
                metrics,
                raw_policy,
                source=source,
            )
        )

        if normalized_reason == "ci-gain-met":
            errors.extend(
                _variance_raw_gain_errors(
                    metrics,
                    raw_policy,
                    source=source,
                )
            )
    return errors


def _variance_raw_status_errors(
    metrics: Mapping[str, Any], *, source: str
) -> list[str]:
    errors: list[str] = []
    if metrics.get("ve_enabled") is not False:
        errors.append(f"{source}.metrics.ve_enabled must be false after restoration.")
    if metrics.get("ve_enabled_during_validation") is not True:
        errors.append(
            f"{source}.metrics.ve_enabled_during_validation=true is required for "
            "ci_gain_met."
        )
    if metrics.get("subject_restored_after_ab") is not True:
        errors.append(
            f"{source}.metrics.subject_restored_after_ab=true is required for "
            "ci_gain_met."
        )
    if metrics.get("met_threshold") is not True:
        errors.append(
            f"{source}.metrics.met_threshold=true is required for ci_gain_met."
        )
    return errors


def _variance_raw_measurement_errors(
    metrics: Mapping[str, Any], *, source: str
) -> tuple[
    list[str], float | None, float | None, float | None, tuple[float, float] | None
]:
    errors: list[str] = []
    ppl_no_ve = _finite_number(metrics.get("ppl_no_ve"))
    ppl_with_ve = _finite_number(metrics.get("ppl_with_ve"))
    ab_gain = _finite_number(metrics.get("ab_gain"))
    if ppl_no_ve is None or ppl_no_ve <= 0.0:
        errors.append(f"{source}.metrics.ppl_no_ve must be finite and positive.")
    if ppl_with_ve is None or ppl_with_ve <= 0.0:
        errors.append(f"{source}.metrics.ppl_with_ve must be finite and positive.")
    if ppl_no_ve is not None and ppl_with_ve is not None and ppl_with_ve >= ppl_no_ve:
        errors.append(
            f"{source}.metrics.ppl_with_ve must improve on ppl_no_ve for ci_gain_met."
        )
    if ab_gain is None:
        errors.append(f"{source}.metrics.ab_gain must be finite for ci_gain_met.")
    elif ppl_no_ve is not None and ppl_no_ve > 0.0 and ppl_with_ve is not None:
        measured_gain = (ppl_no_ve - ppl_with_ve) / ppl_no_ve
        if not math.isclose(ab_gain, measured_gain, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(
                f"{source}.metrics.ab_gain must match the measured PPL improvement."
            )

    ratio_ci = _finite_pair(metrics.get("ratio_ci"))
    if ratio_ci is None:
        errors.append(f"{source}.metrics.ratio_ci must be a finite two-value interval.")
    else:
        ratio_lower, ratio_upper = ratio_ci
        if ratio_lower <= 0.0 or ratio_lower > ratio_upper:
            errors.append(f"{source}.metrics.ratio_ci must be positive and ordered.")
    return errors, ppl_no_ve, ppl_with_ve, ab_gain, ratio_ci


def _variance_raw_calibration_errors(
    metrics: Mapping[str, Any], *, source: str
) -> tuple[list[str], int | None]:
    errors: list[str] = []
    calibration = _mapping(metrics.get("calibration"))
    coverage = (
        _nonnegative_int(calibration.get("coverage"))
        if calibration is not None
        else None
    )
    requested = (
        _nonnegative_int(calibration.get("requested"))
        if calibration is not None
        else None
    )
    if requested is None or requested <= 0:
        errors.append(f"{source}.metrics.calibration.requested must be positive.")
    elif coverage is not None and coverage > requested:
        errors.append(
            f"{source}.metrics.calibration.coverage cannot exceed "
            "calibration.requested."
        )

    windows_used = _nonnegative_int(metrics.get("ab_windows_used"))
    seed_used = _nonnegative_int(metrics.get("ab_seed_used"))
    if windows_used is None or windows_used <= 0:
        errors.append(f"{source}.metrics.ab_windows_used must be positive.")
    elif coverage is not None and windows_used != coverage:
        errors.append(
            f"{source}.metrics.ab_windows_used must match calibration.coverage."
        )
    if seed_used is None:
        errors.append(f"{source}.metrics.ab_seed_used must be an integer.")
    return errors, coverage


def _variance_raw_policy_errors(
    policy: Mapping[str, Any],
    *,
    source: str,
    ppl_no_ve: float | None,
    ppl_with_ve: float | None,
    ab_gain: float | None,
    ratio_ci: tuple[float, float] | None,
) -> list[str]:
    errors: list[str] = []
    min_gain = _finite_number(policy.get("min_gain"))
    deadband = _finite_number(policy.get("tie_breaker_deadband"))
    if min_gain is None or min_gain < 0.0 or deadband is None or deadband < 0.0:
        errors.append(
            f"{source}.policy min_gain and tie_breaker_deadband must be "
            "finite and non-negative."
        )
    elif ab_gain is not None and ab_gain < min_gain + deadband:
        errors.append(f"{source}.metrics.ab_gain does not meet the policy threshold.")
    min_relative_gain = _finite_number(policy.get("min_rel_gain"))
    min_effect = _finite_number(policy.get("min_effect_lognll"))
    absolute_floor = _finite_number(policy.get("absolute_floor_ppl"))
    if min_relative_gain is None or min_relative_gain < 0.0:
        errors.append(f"{source}.policy.min_rel_gain must be non-negative.")
    elif ab_gain is not None and ab_gain < min_relative_gain:
        errors.append(f"{source}.metrics.ab_gain does not meet min_rel_gain.")
    if min_effect is None or min_effect < 0.0:
        errors.append(f"{source}.policy.min_effect_lognll must be non-negative.")
    elif (
        ppl_no_ve is not None
        and ppl_no_ve > 0
        and ppl_with_ve is not None
        and ppl_with_ve > 0
        and math.log(ppl_no_ve) - math.log(ppl_with_ve) < min_effect
    ):
        errors.append(
            f"{source}.metrics PPL improvement does not meet min_effect_lognll."
        )
    if absolute_floor is None or absolute_floor < 0.0:
        errors.append(f"{source}.policy.absolute_floor_ppl must be non-negative.")
    elif (
        ppl_no_ve is not None
        and ppl_with_ve is not None
        and ppl_no_ve - ppl_with_ve < absolute_floor
    ):
        errors.append(
            f"{source}.metrics PPL improvement does not meet absolute_floor_ppl."
        )
    if (
        ratio_ci is not None
        and min_relative_gain is not None
        and min_effect is not None
    ):
        required_upper = min(1.0 - min_relative_gain, math.exp(-min_effect))
        if ratio_ci[1] > required_upper:
            errors.append(
                f"{source}.metrics.ratio_ci does not meet the policy threshold."
            )
    return errors


def _variance_raw_point_errors(
    metrics: Mapping[str, Any],
    *,
    source: str,
    coverage: int | None,
    ppl_no_ve: float | None,
    ppl_with_ve: float | None,
) -> list[str]:
    errors: list[str] = []
    points = _mapping(metrics.get("ab_point_estimates"))
    if points is None:
        errors.append(f"{source}.metrics.ab_point_estimates is required.")
    else:
        if _normalized_token(str(points.get("tag") or "")) != "post-edit":
            errors.append(f"{source}.metrics.ab_point_estimates.tag must be post_edit.")
        if points.get("coverage") != coverage:
            errors.append(
                f"{source}.metrics.ab_point_estimates.coverage must match "
                "calibration.coverage."
            )
        for key, expected in (
            ("ppl_no_ve", ppl_no_ve),
            ("ppl_with_ve", ppl_with_ve),
        ):
            if _finite_number(points.get(key)) != expected:
                errors.append(
                    f"{source}.metrics.ab_point_estimates.{key} must match "
                    f"{source}.metrics.{key}."
                )
    return errors


def _variance_raw_gain_errors(
    metrics: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
    *,
    source: str,
) -> list[str]:
    errors = _variance_raw_status_errors(metrics, source=source)
    measurement_errors, ppl_no_ve, ppl_with_ve, ab_gain, ratio_ci = (
        _variance_raw_measurement_errors(metrics, source=source)
    )
    errors.extend(measurement_errors)
    calibration_errors, coverage = _variance_raw_calibration_errors(
        metrics, source=source
    )
    errors.extend(calibration_errors)
    if policy is not None:
        errors.extend(
            _variance_raw_policy_errors(
                policy,
                source=source,
                ppl_no_ve=ppl_no_ve,
                ppl_with_ve=ppl_with_ve,
                ab_gain=ab_gain,
                ratio_ci=ratio_ci,
            )
        )
    errors.extend(
        _variance_raw_point_errors(
            metrics,
            source=source,
            coverage=coverage,
            ppl_no_ve=ppl_no_ve,
            ppl_with_ve=ppl_with_ve,
        )
    )
    return errors
