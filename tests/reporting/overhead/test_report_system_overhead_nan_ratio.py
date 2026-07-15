from __future__ import annotations

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)
from tests.reporting.overhead._support import (
    overhead_baseline_report,
    overhead_run_report,
)


def _reports_with_sys_overhead_zero_base() -> tuple[dict, dict]:
    report = overhead_run_report(
        metrics={
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            },
            # Provide explicit p50 latency for edited
            "latency_ms_p50": 20.0,
        },
        edit_name="noop",
        tier="balanced",
        profile="dev",
    )
    baseline = overhead_baseline_report(
        metrics={
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            },
            # Explicit p50 latency baseline = 0 → ratio becomes NaN
            "latency_ms_p50": 0.0,
        },
        tier="balanced",
        profile="dev",
    )
    return report, baseline


def test_system_degradation_nan_when_baseline_zero() -> None:
    rep, base = _reports_with_sys_overhead_zero_base()
    cert = make_report(rep, base)
    sys = cert.get("system_overhead", {})
    assert isinstance(sys, dict)
    entry = sys.get("latency_ms_p50") or sys.get("latency_ms_per_tok")
    assert isinstance(entry, dict)
    assert "ratio" not in entry
