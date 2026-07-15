"""Strict evidence reconciliation for the verified no-adjustment variance path."""

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
from .assurance_guard_validation_variance_provenance import (
    _variance_ab_provenance_errors,
    _variance_details_mirror_errors,
    _variance_policy_calibration_errors,
)


def _variance_noop_evidence_errors(
    report: Mapping[str, Any],
    variance: Mapping[str, Any],
    entry: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    source: str,
) -> list[str]:
    errors: list[str] = []
    for top_key, raw_key, expected in (
        ("enabled", "ve_enabled", False),
        ("ve_enabled_during_validation", "ve_enabled_during_validation", False),
        ("subject_restored_after_ab", "subject_restored_after_ab", True),
        ("met_threshold", "met_threshold", False),
    ):
        if (
            variance.get(top_key) is not expected
            or metrics.get(raw_key) is not expected
        ):
            errors.append(
                f"variance.{top_key} and {source}.metrics.{raw_key} must both be "
                f"{str(expected).lower()} for no_adjustment_required."
            )

    top_policy = _mapping(variance.get("policy"))
    raw_policy = _mapping(entry.get("policy"))
    if top_policy is None or raw_policy is None:
        errors.append(
            f"{source}.policy and variance.policy are required for "
            "no_adjustment_required."
        )
    elif top_policy != raw_policy:
        errors.append(f"{source}.policy must match variance.policy exactly.")

    details_errors, details_stats = _variance_details_mirror_errors(
        entry,
        metrics,
        source=source,
    )
    errors.extend(details_errors)
    errors.extend(
        _variance_policy_calibration_errors(raw_policy, metrics, source=source)
    )
    errors.extend(_variance_policy_semantic_errors(raw_policy, metrics, source=source))
    report_errors, expected_provenance = _variance_report_binding_errors(
        report,
        raw_policy,
        source=source,
    )
    errors.extend(report_errors)

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
    if requested is None or requested <= 0 or coverage is None or coverage > requested:
        errors.append(f"{source}.metrics no-adjustment calibration counts are invalid.")

    ppl_no_ve = _finite_number(metrics.get("ppl_no_ve"))
    ppl_with_ve = _finite_number(metrics.get("ppl_with_ve"))
    ab_gain = _finite_number(metrics.get("ab_gain"))
    ratio_ci = _finite_pair(metrics.get("ratio_ci"))
    if (
        ppl_no_ve is None
        or ppl_no_ve <= 0.0
        or ppl_with_ve is None
        or not math.isclose(ppl_no_ve, ppl_with_ve, rel_tol=1e-12, abs_tol=1e-12)
    ):
        errors.append(
            f"{source}.metrics no-adjustment PPL arms must be finite, positive, "
            "and equal."
        )
    if ab_gain is None or not math.isclose(ab_gain, 0.0, abs_tol=1e-12):
        errors.append(f"{source}.metrics.ab_gain must be zero for no adjustment.")
    if ratio_ci is None or ratio_ci != (1.0, 1.0):
        errors.append(
            f"{source}.metrics.ratio_ci must equal [1.0, 1.0] for no adjustment."
        )

    points = _mapping(metrics.get("ab_point_estimates"))
    if points is None:
        errors.append(f"{source}.metrics.ab_point_estimates is required.")
    else:
        if _normalized_token(str(points.get("tag") or "")) != "post-edit":
            errors.append(f"{source}.metrics.ab_point_estimates.tag must be post_edit.")
        for key, expected_ppl in (
            ("ppl_no_ve", ppl_no_ve),
            ("ppl_with_ve", ppl_with_ve),
        ):
            if _finite_number(points.get(key)) != expected_ppl:
                errors.append(
                    f"{source}.metrics.ab_point_estimates.{key} must match "
                    f"{source}.metrics.{key}."
                )
        if points.get("coverage") != coverage:
            errors.append(
                f"{source}.metrics.ab_point_estimates.coverage must match "
                "calibration.coverage."
            )

    windows_used = _nonnegative_int(metrics.get("ab_windows_used"))
    if windows_used is None or windows_used <= 0 or windows_used != coverage:
        errors.append(
            f"{source}.metrics.ab_windows_used must be a positive integer equal "
            "to calibration.coverage."
        )

    for top_key, raw_key in (
        ("gain", "ab_gain"),
        ("ppl_no_ve", "ppl_no_ve"),
        ("ppl_with_ve", "ppl_with_ve"),
        ("ratio_ci", "ratio_ci"),
    ):
        if top_key == "ratio_ci":
            matches = _finite_pair(variance.get(top_key)) == _finite_pair(
                metrics.get(raw_key)
            )
        else:
            matches = variance.get(top_key) == metrics.get(raw_key)
        if top_key not in variance or not matches:
            errors.append(
                f"variance.{top_key} must match {source}.metrics.{raw_key} exactly."
            )

    ab_test = _mapping(variance.get("ab_test"))
    if ab_test is None:
        errors.append(
            "variance.ab_test is required for no_adjustment_required assurance."
        )
        top_provenance = None
    else:
        if ab_test.get("seed") != metrics.get("ab_seed_used"):
            errors.append(
                f"variance.ab_test.seed must match {source}.metrics.ab_seed_used."
            )
        if ab_test.get("windows_used") != metrics.get("ab_windows_used"):
            errors.append(
                f"variance.ab_test.windows_used must match "
                f"{source}.metrics.ab_windows_used."
            )
        if ab_test.get("point_estimates") != metrics.get("ab_point_estimates"):
            errors.append(
                f"variance.ab_test.point_estimates must match "
                f"{source}.metrics.ab_point_estimates."
            )
        top_provenance = _mapping(ab_test.get("provenance"))
        raw_provenance = _mapping(metrics.get("ab_provenance"))
        for condition in ("condition_a", "condition_b"):
            if (
                top_provenance is None
                or raw_provenance is None
                or top_provenance.get(condition) != raw_provenance.get(condition)
            ):
                errors.append(
                    f"variance.ab_test.provenance.{condition} must match raw evidence."
                )

    errors.extend(
        _variance_ab_provenance_errors(
            metrics,
            coverage,
            source=source,
            top_provenance=top_provenance,
            details_stats=details_stats,
            condition_b_statuses=frozenset({"no-scales"}),
            expected_provenance=expected_provenance,
        )
    )
    errors.extend(
        _variance_measurement_errors(
            variance,
            entry,
            metrics,
            coverage,
            raw_policy,
            source=source,
            no_adjustment=True,
        )
    )
    return errors
