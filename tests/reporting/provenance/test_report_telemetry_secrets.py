from __future__ import annotations

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def _mk_report() -> tuple[dict, dict]:
    r = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 1,
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev"},
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {"name": "structured"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 50.0,
                "final": 50.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": (1.0, 1.0),
            },
            "preview_total_tokens": 1000,
            "final_total_tokens": 1000,
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [1.0, 1.1],
                "token_counts": [100, 200],
            },
            "final": {
                "window_ids": [3, 4],
                "logloss": [1.0, 1.1],
                "token_counts": [100, 200],
            },
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    b = {
        **r,
        "run_id": "baseline-1",
        "edit": {"name": "noop"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 50.0,
                "final": 50.0,
            },
            "bootstrap": {"replicates": 200, "alpha": 0.05},
        },
    }
    return r, b


def test_telemetry_summary_contains_no_paths_or_usernames():
    r, b = _mk_report()
    cert = make_report(r, b)
    s = (cert.get("telemetry", {}) or {}).get("summary_line", "")
    assert isinstance(s, str)
    # No obvious path separators or at-signs
    assert "/" not in s and "\\" not in s and "@" not in s
