from invarlock.reporting.certificate import make_certificate


def test_certificate_raises_on_drift_ratio_inconsistency():
    report = {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_gpt2",
            "device": "cpu",
            "ts": "2025-01-01T00:00:00",
            "commit": "dead",
            "seed": 42,
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 2,
            "final_n": 2,
        },
        "edit": {
            "name": "structured",
            "plan_digest": "abcd",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.2,
                "ratio_vs_baseline": 1.02,
            },
            # Paired delta summary mean mismatched intentionally
            "paired_delta_summary": {"mean": 0.1, "degenerate": False},
        },
        "evaluation_windows": {
            "final": {"window_ids": [1, 2], "logloss": [1.0, 2.0]},
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    baseline = {
        "run_id": "b",
        "model_id": "gpt2",
        "evaluation_windows": {"final": {"window_ids": [2, 1], "logloss": [2.0, 1.0]}},
    }

    # After normalization, this inconsistency is tolerated; certificate still returned
    report.setdefault("metrics", {}).setdefault("window_plan", {})["profile"] = "ci"
    cert = make_certificate(report, baseline)
    assert isinstance(cert, dict)
