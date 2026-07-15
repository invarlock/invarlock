from __future__ import annotations

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)
from tests.reporting.overhead._support import overhead_run_report


def _mk_report_with_metric_impact(metrics: dict) -> dict:
    return overhead_run_report(
        metrics={
            # primary metric and overhead metrics
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": (1.0, 1.0),
            },
            # overhead metrics
            **metrics,
        },
        edit_name="noop",
        tier="balanced",
        profile="dev",
    )


def test_system_overhead_json_keys():
    report = _mk_report_with_metric_impact(
        {"latency_ms_p50": 2.0, "latency_ms_p95": 3.5, "throughput_sps": 77.7}
    )
    baseline = _mk_report_with_metric_impact(
        {"latency_ms_p50": 1.5, "latency_ms_p95": 3.0, "throughput_sps": 80.0}
    )
    cert = make_report(report, baseline)
    so = cert.get("system_overhead", {})
    assert isinstance(so, dict)
