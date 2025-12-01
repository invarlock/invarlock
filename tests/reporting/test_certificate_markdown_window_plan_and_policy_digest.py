from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_markdown_window_plan_line_and_policy_digest_changed():
    # Build a valid certificate via make_certificate
    report = {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
        "data": {
            "dataset": "prov",
            "split": "val",
            "seq_len": 128,
            "stride": 8,
            "preview_n": 1,
            "final_n": 2,
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
    # Inject window_plan and policy_digest for rendering branches
    cert["window_plan"] = {"profile": "ci", "actual_preview": 1, "actual_final": 2}
    cert["policy_digest"] = {
        "policy_version": "policy-v1",
        "thresholds_hash": "a" * 64,
        "changed": True,
    }

    md = render_certificate_markdown(cert)
    # Window plan one-liner includes seq_len and uses actual_* fields
    assert "Window Plan: ci, 1/2, seq_len=128" in md.replace("  ", " ")
    # Policy digest shows version and shortened thresholds hash and change note
    assert "**Policy Version:** policy-v1" in md
    assert "Thresholds Digest:" in md and "aaaaaaaa" in md
