from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_guard_overhead_row_na_and_hysteresis_note():
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
        **report,
        "edit": {
            "name": "baseline",
            "plan_digest": "baseline_noop",
            "deltas": {"params_changed": 0},
        },
    }
    cert = make_certificate(report, baseline)
    # Force guard overhead evaluated without measured fields
    cert["guard_overhead"] = {"evaluated": True, "overhead_threshold": 0.012}
    cert.setdefault("validation", {})["guard_overhead_acceptable"] = True
    cert["validation"]["hysteresis_applied"] = True
    md = render_certificate_markdown(cert)
    assert "Guard Overhead Acceptable" in md and "N/A" in md
    assert "hysteresis applied" in md
