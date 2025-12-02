from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_spectral_multiple_testing_family_count_computed_when_missing_m():
    # Minimal report with spectral guard policy providing family_caps and multiple_testing without m
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
            {
                "name": "spectral",
                "policy": {
                    "family_caps": {"ffn": {"kappa": 2.5}, "attn": {"kappa": 2.8}},
                    "multiple_testing": {"method": "bh", "alpha": 0.05},  # m omitted
                },
                "metrics": {},  # leave families empty so caps keys drive the count
            }
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
    # Expect the helper to compute m from families_present (from family_caps keys)
    assert spectral.get("bh_family_count") == 2
