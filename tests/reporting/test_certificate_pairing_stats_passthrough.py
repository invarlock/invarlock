from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def _mk_report_with_stats() -> tuple[dict, dict]:
    report = {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
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
                "display_ci": (1.0, 1.0),
            },
            "stats": {
                "requested_preview": 2,
                "requested_final": 2,
                "actual_preview": 2,
                "actual_final": 2,
                "coverage_ok": True,
            },
            "window_plan": {"profile": "ci", "preview_n": 2, "final_n": 2},
            "bootstrap": {"replicates": 10, "alpha": 0.05},
            # Also set window match/overlap to ensure stat propagation
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1, 2],
                "logloss": [2.30, 2.31],
                "token_counts": [1, 1],
            },
            "final": {
                "window_ids": [3, 4],
                "logloss": [2.30, 2.31],
                "token_counts": [1, 1],
            },
        },
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    baseline = {
        "meta": {"auto": {"tier": "balanced"}},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    return report, baseline


def test_pairing_stats_passthrough_into_dataset_windows_stats() -> None:
    report, base = _mk_report_with_stats()
    cert = make_certificate(report, base)
    ds = cert.get("dataset", {})
    windows = ds.get("windows", {}) if isinstance(ds, dict) else {}
    stats = windows.get("stats", {}) if isinstance(windows, dict) else {}
    # Ensure match/overlap and pairing present (coverage may be absent on tiny runs)
    assert "window_match_fraction" in stats and "window_overlap_fraction" in stats
    assert "pairing" in stats
