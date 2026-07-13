import math

from invarlock.reporting.report_metric_impact import prepare_guard_metric_impact_section
from invarlock.reporting.validation.report import compute_validation_flags


def test_guard_metric_impact_prepare_rejects_string_limit():
    raw = {
        "metric_kind": "ppl_causal",
        "direction": "lower",
        "degradation_basis": "relative_increase",
        "bare_value": 100.0,
        "guarded_value": 101.5,
        "degradation": 0.015,
        "degradation_limit": "0.02",  # string convertible
        "display_value": 1.5,
        "display_unit": "percent",
        "bare_facts": {"weighted_logloss_sum": math.log(100.0), "token_count": 1},
        "guarded_facts": {
            "weighted_logloss_sum": math.log(101.5),
            "token_count": 1,
        },
        "checks": {"guard_metric_impact": True},
    }
    sanitized, passed = prepare_guard_metric_impact_section(raw)
    assert sanitized["evaluated"] is False
    assert passed is False
    assert sanitized["diagnostics"]
    assert sanitized["diagnostics"][0]["severity"] == "error"

    flags = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={"caps_applied": 0, "max_caps": 5},
        rmt={"stable": True},
        invariants={"status": "pass"},
        guard_metric_impact=sanitized,
    )
    assert flags["guard_metric_impact_acceptable"] is False


def test_guard_metric_impact_prepare_invalid_degradation_and_limit():
    raw = {"degradation": float("nan"), "degradation_limit": "bad"}
    sanitized, passed = prepare_guard_metric_impact_section(raw)
    # Invalid degradation and limit are not evidence of a passing gate.
    assert sanitized["evaluated"] is False and passed is False
    assert sanitized["diagnostics"]
    assert sanitized["diagnostics"][0]["severity"] == "error"
    flags = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={"caps_applied": 0, "max_caps": 5},
        rmt={"stable": True},
        invariants={"status": "pass"},
        guard_metric_impact=sanitized,
    )
    assert flags["guard_metric_impact_acceptable"] is False
