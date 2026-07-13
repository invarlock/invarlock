from __future__ import annotations

from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)
from tests.reporting.overhead._support import (
    overhead_baseline_report,
    overhead_run_report,
)


def _mk_report_latency_fallback() -> dict:
    return overhead_run_report(
        metrics={
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            },
            # fallback-only key for edited
            "latency_ms_per_tok": 11.0,
            "throughput_tok_per_s": 100.0,
        },
        edit_name="noop",
        tier="balanced",
        profile="dev",
    )


def _mk_baseline_explicit() -> dict:
    return overhead_baseline_report(
        metrics={
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            },
            "latency_ms_p50": 10.0,
            "throughput_sps": 120.0,
        },
        tier="balanced",
        profile="dev",
    )


def test_system_overhead_sources_mixed_and_markdown_na() -> None:
    rep = _mk_report_latency_fallback()
    base = _mk_baseline_explicit()
    cert = make_report(rep, base)
    sys = cert.get("system_overhead", {})
    # Keys should include p50 latency entry with ratio; throughput may be absent on tiny runs
    assert "latency_ms_p50" in sys
    md = render_report_markdown(cert)
    # Ensure the section renders; check presence of latency row
    assert "System Overhead" in md and "Latency p50" in md


def test_system_overhead_does_not_reuse_edited_fallback_for_baseline() -> None:
    rep = _mk_report_latency_fallback()
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

    latency_entry = cert.get("system_overhead", {}).get("latency_ms_p50", {})
    assert latency_entry.get("edited") == 11.0
    assert "baseline" not in latency_entry
