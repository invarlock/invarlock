from __future__ import annotations

from invarlock.reporting.report_overhead import prepare_guard_overhead_section
from invarlock.reporting.report_provenance import compute_report_digest


def test_compute_report_digest_minimal():
    rep = {
        "meta": {"model_id": "m", "adapter": "hf", "commit": "abc", "ts": "t"},
        "edit": {"name": "noop", "plan_digest": "deadbeef"},
        "metrics": {"spectral": {"caps_applied": 0}, "rmt": {"outliers": 0}},
    }
    h = compute_report_digest(rep)
    assert isinstance(h, str) and len(h) == 16


def test_prepare_guard_overhead_threshold_boundary():
    # Ratio equals 1 + threshold should PASS
    payload = {"bare_ppl": 100.0, "guarded_ppl": 101.5, "overhead_threshold": 0.015}
    out, passed = prepare_guard_overhead_section(payload)
    assert out.get("evaluated") is True and passed is True
    # Diagnostics should flow through unchanged
    payload2 = {
        "bare_ppl": 100.0,
        "guarded_ppl": 101.5,
        "overhead_threshold": 0.015,
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
    out2, _ = prepare_guard_overhead_section(payload2)
    assert out2["diagnostics"][0]["message"] == "note"
    assert out2["diagnostics"][1]["severity"] == "warning"
