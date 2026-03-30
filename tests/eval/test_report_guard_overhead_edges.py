import math

from invarlock.reporting.report_overhead import prepare_guard_overhead_section
from invarlock.reporting.report_validation import compute_validation_flags


def test_guard_overhead_prepare_with_ratio_and_string_threshold():
    raw = {
        "overhead_ratio": 1.015,
        "overhead_threshold": "0.02",  # string convertible
        "messages": ["ok"],
        "warnings": None,
        "errors": None,
        "checks": None,
    }
    sanitized, passed = prepare_guard_overhead_section(raw)
    assert sanitized["evaluated"] is True
    assert passed is True
    assert math.isclose(sanitized["overhead_ratio"], 1.015, rel_tol=1e-9)
    assert math.isclose(sanitized["threshold_percent"], 2.0, rel_tol=1e-9)

    flags = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={"caps_applied": 0, "max_caps": 5},
        rmt={"stable": True},
        invariants={"status": "pass"},
        guard_overhead=sanitized,
    )
    assert flags["guard_overhead_acceptable"] is True


def test_guard_overhead_prepare_invalid_ratio_and_threshold():
    raw = {"overhead_ratio": float("nan"), "overhead_threshold": "bad"}
    sanitized, passed = prepare_guard_overhead_section(raw)
    # Invalid ratio and threshold should result in not evaluated but soft-pass
    assert sanitized["evaluated"] is False and passed is True
    assert sanitized["diagnostics"]
    assert sanitized["diagnostics"][0]["severity"] == "warning"
    flags = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={"caps_applied": 0, "max_caps": 5},
        rmt={"stable": True},
        invariants={"status": "pass"},
        guard_overhead=sanitized,
    )
    # In PM-only policy, missing overhead metrics do not fail validation
    assert flags["guard_overhead_acceptable"] is True
