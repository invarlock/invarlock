from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_spectral_families_from_family_stats_only():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
        },
        "guards": [
            {
                "name": "spectral",
                "metrics": {
                    "family_stats": {
                        "ffn": {
                            "count": 3,
                            "mean": 1.2,
                            "std": 0.1,
                            "min": 1.0,
                            "max": 1.3,
                        },
                        "attn": {
                            "count": 2,
                            "mean": 1.1,
                            "std": 0.1,
                            "min": 1.0,
                            "max": 1.2,
                        },
                    }
                },
                "policy": {},
            }
        ],
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "structured",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)
    spectral = cert.get("spectral", {})
    assert spectral.get("families", {}).get("ffn", {}).get("count") == 3
