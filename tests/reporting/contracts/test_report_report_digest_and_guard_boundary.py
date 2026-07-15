from __future__ import annotations

from invarlock.reporting.report_metric_impact import prepare_guard_metric_impact_section
from invarlock.reporting.report_provenance import compute_report_digest
from tests.reporting._support_guard_metric_impact import ppl_guard_context


def test_compute_report_digest_minimal():
    rep = {
        "meta": {"model_id": "m", "adapter": "hf", "commit": "abc", "ts": "t"},
        "edit": {"name": "noop", "plan_digest": "deadbeef"},
        "metrics": {"spectral": {"caps_applied": 0}, "rmt": {"outliers": 0}},
    }
    h = compute_report_digest(rep)
    assert isinstance(h, str) and len(h) == 64


def test_prepare_guard_metric_degradation_limit_boundary():
    # Ratio equals 1 + threshold should PASS
    payload = ppl_guard_context(100.0, 101.5, degradation_limit=0.015)
    out, passed = prepare_guard_metric_impact_section(payload)
    assert out.get("evaluated") is True and passed is True
    # Caller-supplied diagnostics are not trusted over validator diagnostics.
    payload2 = {
        **ppl_guard_context(100.0, 101.5, degradation_limit=0.015),
        "diagnostics": [
            {
                "kind": "validation_info",
                "severity": "info",
                "message": "note",
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
    out2, _ = prepare_guard_metric_impact_section(payload2)
    assert all(item["message"] != "note" for item in out2["diagnostics"])
    assert any("PASSED" in item["message"] for item in out2["diagnostics"])
