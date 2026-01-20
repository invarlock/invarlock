from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_certificate_ratio_ci_source_run_metrics_on_wrong_type_return():
    """
    If compute_paired_delta_log_ci returns a wrong type (non-tuple),
    the conversion to ratio CI should fail inside the try-block and
    the code must fall back to using run_metrics (ratio_ci_source="run_metrics").
    """
    report = {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
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
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            },
            "logloss_preview_ci": (2.2, 2.4),
            "logloss_final_ci": (2.2, 2.4),
            "logloss_delta_ci": (-0.1, 0.1),
            "bootstrap": {"method": "bca", "replicates": 10, "alpha": 0.1, "seed": 0},
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [1.0, 2.0]}},
    }
    baseline = {
        "run_id": "b",
        "model_id": "gpt2",
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [1.0, 2.0]}},
    }

    # Return a wrong type that doesn't raise immediately from the patched function,
    # but will cause conversion (logspace_to_ratio_ci) to raise inside the try.
    with patch(
        "invarlock.reporting.certificate.compute_paired_delta_log_ci",
        return_value="bad-type",
    ):
        cert = make_certificate(report, baseline)
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    assert stats.get("pairing") == "run_metrics"
