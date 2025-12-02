from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_paired_path_skips_ci_when_zero_replicates():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.2,
            "ppl_ratio": 1.02,
            "ppl_preview_ci": (9.5, 10.5),
            "ppl_final_ci": (9.7, 10.7),
            "ppl_ratio_ci": (0.98, 1.06),
            "bootstrap": {
                "method": "percentile",
                "replicates": 0,
                "alpha": 0.1,
                "seed": 0,
            },
            # Provide a paired delta summary to avoid degenerate paths
            "paired_delta_summary": {"mean": 0.0198, "degenerate": False},
        },
        "data": {
            "dataset": "dummy",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 2,
            "final_n": 2,
        },
        "guards": [],
        "edit": {
            "name": "structured",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]}},
    }
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "ppl_final": 9.8,
        "evaluation_windows": {
            "final": {"window_ids": [1, 2], "logloss": [0.09, 0.19]}
        },
    }

    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    # Paired windows detected, but with zero replicates we keep ratio_ci_source as run_metrics
    assert stats.get("paired_windows") == 2
    assert stats.get("pairing") == "run_metrics"
