from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_spectral_stability_score_from_generic_key():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            # Provide spectral guard metrics including generic stability_score
            "spectral": {
                "max_spectral_norm": 2.0,
                "mean_spectral_norm": 1.0,
                "stability_score": 0.77,
            },
        },
        "guards": [{"name": "spectral", "metrics": {"stability_score": 0.77}}],
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
    assert summary.get("stability_score") == 0.77
