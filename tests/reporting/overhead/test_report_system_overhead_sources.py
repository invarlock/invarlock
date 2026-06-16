from __future__ import annotations

from invarlock.reporting.render import render_report_markdown
from invarlock.reporting.report_make import make_report


def _mk_report_latency_fallback() -> dict:
    return {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "metrics": {
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
        "evaluation_windows": {
            "preview": {
                "window_ids": [1],
                "logloss": [2.302585093],
                "token_counts": [1],
            },
            "final": {"window_ids": [2], "logloss": [2.302585093], "token_counts": [1]},
        },
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def _mk_baseline_explicit() -> dict:
    return {
        "meta": {"auto": {"tier": "balanced"}},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
            "latency_ms_p50": 10.0,
            "throughput_sps": 120.0,
        },
    }


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
    base = {
        "meta": {"auto": {"tier": "balanced"}},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }

    cert = make_report(rep, base)

    latency_entry = cert.get("system_overhead", {}).get("latency_ms_p50", {})
    assert latency_entry.get("edited") == 11.0
    assert "baseline" not in latency_entry
