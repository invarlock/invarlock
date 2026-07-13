from __future__ import annotations

import math

from invarlock.reporting import report_metric_impact as report_metric_impact_mod


def test_prepare_guard_metric_impact_section_direct_measurement_and_lists() -> None:
    raw = {
        "degradation_limit": "0.02",
        "metric_kind": "ppl_causal",
        "direction": "lower",
        "degradation_basis": "relative_increase",
        "bare_value": 10.0,
        "guarded_value": 10.5,
        "degradation": 0.05,
        "display_value": 5.0,
        "display_unit": "percent",
        "bare_facts": {"weighted_logloss_sum": math.log(10.0), "token_count": 1},
        "guarded_facts": {
            "weighted_logloss_sum": math.log(10.5),
            "token_count": 1,
        },
        "checks": {"guard_metric_impact": True},
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
    sanitized, passed = report_metric_impact_mod.prepare_guard_metric_impact_section(
        raw
    )
    assert isinstance(sanitized, dict)
    assert sanitized["degradation_limit"] == 0.01
    assert sanitized["evaluated"] is False
    assert sanitized["diagnostics"][0]["severity"] == "error"
    assert passed is False
