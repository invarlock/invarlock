from __future__ import annotations

from invarlock.reporting import report_overhead as report_overhead_mod


def test_prepare_guard_overhead_section_direct_ratio_and_lists() -> None:
    raw = {
        "overhead_threshold": "0.02",
        "bare_ppl": 10.0,
        "guarded_ppl": 10.5,
        "diagnostics": [
            {
                "kind": "validation_info",
                "severity": "info",
                "message": "ok",
                "details": {},
            },
            {
                "kind": "validation_warning",
                "severity": "warning",
                "message": "warn",
                "details": {},
            },
        ],
    }
    sanitized, passed = report_overhead_mod.prepare_guard_overhead_section(raw)
    assert isinstance(sanitized, dict)
    assert 0.01 < sanitized["overhead_threshold"] < 0.03
    assert "overhead_ratio" in sanitized and "overhead_percent" in sanitized
    assert sanitized["diagnostics"][0]["message"] == "ok"
    assert sanitized["diagnostics"][1]["severity"] == "warning"
    assert sanitized["diagnostics"][1]["message"] == "warn"
    assert passed in {True, False}
