from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_drift_ci_computed_from_preview_and_final_ci():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.5,
                "ratio_vs_baseline": 10.5 / 10.0,
                "display_ci": (1.0, 1.05),
            },
        },
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
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
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)
    pm = cert.get("primary_metric", {})
    dci = pm.get("display_ci") if isinstance(pm, dict) else None
    assert isinstance(dci, list | tuple) and len(dci) == 2
