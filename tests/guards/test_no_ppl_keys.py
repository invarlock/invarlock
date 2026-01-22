from invarlock.reporting.certificate import make_certificate


def _minimal_pm_report():
    return {
        "meta": {"model_id": "m", "adapter": "hf_causal", "seed": 7, "device": "cpu"},
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "x",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": (10.0, 10.0),
            },
            "bootstrap": {"replicates": 150, "alpha": 0.05, "method": "percentile"},
        },
        "evaluation_windows": {
            "final": {"window_ids": [1], "logloss": [1.0], "token_counts": [100]}
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def _baseline():
    return {
        "run_id": "baseline",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {
            "final": {"window_ids": [1], "logloss": [1.0], "token_counts": [100]}
        },
        "metrics": {
            "bootstrap": {"replicates": 150, "alpha": 0.05, "method": "percentile"}
        },
    }


def test_certificate_has_no_ppl_keys():
    cert = make_certificate(_minimal_pm_report(), _baseline())
    assert "ppl" not in cert
