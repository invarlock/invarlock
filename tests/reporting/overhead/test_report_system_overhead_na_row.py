from __future__ import annotations

from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)
from tests.reporting.overhead._support import (
    overhead_baseline_report,
    overhead_run_report,
)


def test_system_overhead_na_row_when_both_zero() -> None:
    rep = overhead_run_report(
        metrics={
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            }
        },
        edit_name="noop",
        tier="balanced",
        profile="dev",
    )
    base = overhead_baseline_report(
        metrics={
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            }
        },
        tier="balanced",
        profile="dev",
    )
    cert = make_report(rep, base)
    # Force N/A path: both baseline and edited zero for throughput
    cert["system_overhead"] = {"throughput_sps": {"baseline": 0.0, "edited": 0.0}}
    md = render_report_markdown(cert)
    # Row must include N/A columns
    assert "System Overhead" in md and "Throughput (samples/s)" in md and "N/A" in md
