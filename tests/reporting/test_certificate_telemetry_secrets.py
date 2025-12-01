from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def _mk_report() -> tuple[dict, dict]:
    r = {
        "meta": {"model_id": "m", "adapter": "hf_gpt2", "device": "cpu", "seed": 1},
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
        "run_id": "baseline-1",
        "model_id": "m",
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 50.0},
            "bootstrap": {"replicates": 200, "alpha": 0.05},
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
    return r, b


def test_telemetry_summary_contains_no_paths_or_usernames():
    r, b = _mk_report()
    cert = make_certificate(r, b)
    s = (cert.get("telemetry", {}) or {}).get("summary_line", "")
    assert isinstance(s, str)
    # No obvious path separators or at-signs
    assert "/" not in s and "\\" not in s and "@" not in s
