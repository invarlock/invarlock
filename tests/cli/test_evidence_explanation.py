"""Presentation checks use policy bounds, not point estimates or one generic CI."""

from __future__ import annotations

import pytest

from invarlock.evidence_explanation import core_policy_checks


def test_all_configured_checks_survive_a_relative_failure():
    report = {
        "comparison": {"kind": "exact_match_delta_pp", "minimum": -10, "value": -2},
        "uncertainty": {
            "scope": "paired_binary_outcomes",
            "lower": -10.4954435896,
            "upper": 3.5,
        },
        "sample_qualification": {
            "record_count": {"observed": 50, "minimum": 50, "passed": True},
            "interval_width": {
                "observed": 13.9954435896,
                "maximum": 20,
                "unit": "percentage_points",
                "passed": True,
            },
        },
        "side_accuracy": {
            "minimum": 0.99,
            "baseline": {"observed": 1.0, "passed": True},
            "subject": {"observed": 0.98, "passed": False},
        },
    }
    checks = core_policy_checks(report)
    assert [c["passed"] for c in checks] == [False, True, True, True, False]
    assert checks[0]["observed"] == "-10.4954 pp"
    assert checks[0]["required"] == ">= -10 pp"
    assert checks[3]["observed"] == "100%"
    assert checks[4]["observed"] == "98%"
    assert all(
        set(c) == {"name", "observed", "required", "passed", "explanation"}
        for c in checks
    )


def test_extension_interval_is_finite_schedule_and_boundary_is_inclusive():
    checks = core_policy_checks(
        {
            "comparison": {"kind": "scorer_extension_delta_pp", "minimum": -5},
            "uncertainty": {"scope": "authenticated_schedule", "lower": -5, "upper": 6},
        }
    )
    assert checks[0]["name"] == "Finite-schedule lower bound"
    assert checks[0]["passed"] is True
    assert "confidence" not in str(checks)


def test_normalized_loss_uses_upper_ratio_and_ratio_precision():
    checks = core_policy_checks(
        {
            "comparison": {"kind": "normalized_nll_ratio", "maximum": 1.05},
            "uncertainty": {
                "scope": "authenticated_schedule",
                "lower": 0.99,
                "upper": 1.06,
            },
            "sample_qualification": {
                "record_count": {"observed": 50, "minimum": 60, "passed": False},
                "interval_width": {
                    "observed": 0.07,
                    "maximum": 0.05,
                    "unit": "ratio",
                    "passed": False,
                },
            },
        }
    )
    assert checks[0]["observed"] == "1.06"
    assert checks[0]["required"] == "<= 1.05"
    assert checks[2]["observed"] == "0.07 ratio"
    assert not any(c["passed"] for c in checks)


def test_unsupported_comparison_has_no_generic_success_fallback():
    with pytest.raises(ValueError, match="unsupported"):
        core_policy_checks({"comparison": {"kind": "unknown"}, "uncertainty": {}})
