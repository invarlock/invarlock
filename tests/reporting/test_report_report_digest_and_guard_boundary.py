from __future__ import annotations

from invarlock.reporting.report_make import _compute_report_digest
from invarlock.reporting.report_overhead import prepare_guard_overhead_section


def test_compute_report_digest_minimal():
    rep = {
        "meta": {"model_id": "m", "adapter": "hf", "commit": "abc", "ts": "t"},
        "edit": {"name": "noop", "plan_digest": "deadbeef"},
        "metrics": {"spectral": {"caps_applied": 0}, "rmt": {"outliers": 0}},
    }
    h = _compute_report_digest(rep)
    assert isinstance(h, str) and len(h) == 16


def test_prepare_guard_overhead_threshold_boundary():
    # Ratio equals 1 + threshold should PASS
    payload = {"bare_ppl": 100.0, "guarded_ppl": 101.5, "overhead_threshold": 0.015}
    out, passed = prepare_guard_overhead_section(payload)
    assert out.get("evaluated") is True and passed is True
    # Messages/warnings/errors coercion should produce lists
    payload2 = {
        "bare_ppl": 100.0,
        "guarded_ppl": 101.5,
        "overhead_threshold": 0.015,
        "messages": ["note"],
        "warnings": ["warn"],
        "errors": [],
    }
    out2, _ = prepare_guard_overhead_section(payload2)
    assert isinstance(out2.get("messages"), list) and isinstance(
        out2.get("warnings"), list
    )
