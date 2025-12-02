from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def test_make_certificate_paired_baseline_ratio_ci():
    # Prepare subject and baseline with matching window_ids so paired_baseline path is used
    rep = {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "ts": "2024-01-01T00:00:00",
            "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        },
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
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 0,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 4.0,
                "final": 4.0,
                "ratio_vs_baseline": 1.0,
            },
            "bootstrap": {
                "replicates": 10,
                "alpha": 0.05,
                "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
            },
        },
        "evaluation_windows": {
            "final": {
                "window_ids": [1, 2],
                "logloss": [4.0, 4.0],
                "token_counts": [100, 100],
            }
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    base = {
        "meta": {"auto": {"tier": "balanced"}},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 4.0}},
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [4.0, 4.0]}},
    }
    cert = make_certificate(rep, base)
    pm = cert.get("primary_metric", {})
    # Expect a display_ci to be present regardless of method choice
    dci = pm.get("display_ci") if isinstance(pm, dict) else None
    assert isinstance(dci, list | tuple) and len(dci) == 2
