from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_spectral_precedence_from_tier_defaults_no_tables():
    # Report with minimal spectral info; rely on tier defaults for caps/quantile
    report = {
        "meta": {"model_id": "m", "seed": 1, "auto": {"tier": "balanced"}},
        "metrics": {"ppl_preview": 10.0, "ppl_final": 10.0},
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [
            {"name": "spectral", "policy": {}, "metrics": {}},
        ],
        "edit": {
            "name": "structured",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
                "sparsity": None,
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
    # family_z_quantiles/top_z_scores likely empty; sigma_quantile and family_caps provided
    assert spectral.get("sigma_quantile")
    assert spectral.get("family_caps")
