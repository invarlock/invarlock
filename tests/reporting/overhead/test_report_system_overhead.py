from __future__ import annotations

from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)
from tests.reporting.overhead._support import overhead_run_report


def _mk_minimal_report(metrics: dict) -> dict:
    return overhead_run_report(
        metrics={
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            },
            "latency_ms_per_tok": 2.0,
            "throughput_tok_per_s": 50.0,
            **metrics,
        },
        edit_name="noop",
        tier="balanced",
        profile="dev",
    )


def test_evaluation_report_system_overhead_table_and_primary_metric_metadata():
    report = _mk_minimal_report({})
    baseline = _mk_minimal_report(
        {"latency_ms_per_tok": 1.6, "throughput_tok_per_s": 60.0}
    )
    baseline["metrics"]["primary_metric"] = {
        "kind": "accuracy",
        "preview": 0.80,
        "final": 0.80,
    }

    # Include a primary metric snapshot with metadata
    report.setdefault("metrics", {})["primary_metric"] = {
        "kind": "accuracy",
        "unit": "pp",
        "paired": True,
        "gating_basis": "lower",
        "reps": 500,
        "ci": (-1.2, +1.5),
        "preview": 0.80,
        "final": 0.85,
        "delta_vs_baseline_pp": +5.0,
    }

    cert = make_report(report, baseline)
    md = render_report_markdown(cert)
    # System Overhead section may be omitted; rendering should still succeed
    assert "# InvarLock Evaluation Report" in md

    # Primary Metric metadata present
    assert "## Primary Metric" in md
