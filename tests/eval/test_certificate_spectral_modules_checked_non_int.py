from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_spectral_summary_modules_checked_non_int_ignored():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "spectral": {
                "max_spectral_norm": 1.2,
                "mean_spectral_norm": 1.0,
                "modules_checked": "not-an-int",
            },
        },
        "guards": [
            {
                "name": "spectral",
                "metrics": {"max_spectral_norm": 1.2, "mean_spectral_norm": 1.0},
                "policy": {"deadband": "0.10"},  # ensure deadband float coercion path
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
    summary = cert.get("spectral", {}).get("summary", {})
    # modules_checked non-int is ignored; and deadband gets parsed
    assert "modules_checked" not in summary
    assert isinstance(summary.get("deadband"), float)
