from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_markdown_guard_overhead_percent_format():
    report = {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 1,
            "final_n": 1,
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            },
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1],
                "logloss": [2.302585093],
                "token_counts": [1],
            },
            "final": {"window_ids": [2], "logloss": [2.302585093], "token_counts": [1]},
        },
        "edit": {"name": "noop"},
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m"},
        "metrics": {"ppl_final": 10.0},
    }
    cert = make_certificate(report, baseline)
    # Set guard overhead with explicit ratio and threshold; evaluated True
    cert["guard_overhead"] = {
        "evaluated": True,
        "overhead_ratio": 1.015,
        "overhead_threshold": 0.02,
    }
    cert.setdefault("validation", {})["guard_overhead_acceptable"] = True
    md = render_certificate_markdown(cert)
    # Should render a ratio when percent not precomputed
    assert "Guard Overhead Acceptable" in md and "1.015x" in md
