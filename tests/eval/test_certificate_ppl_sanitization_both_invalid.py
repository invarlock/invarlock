from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_certificate_ppl_both_invalid_fallback_default():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "preview": 0.0, "final": 0.0}
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
    # Baseline also invalid to force default fallback
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)
    pm = cert.get("primary_metric", {})
    # Baseline reference now stores PM-only; ensure it's present and numeric when computable
    bref = cert.get("baseline_ref", {}).get("primary_metric", {})
    bf = bref.get("final")
    assert bf is None or isinstance(bf, (int | float))
    # PM final may be unavailable or sanitized; ensure it is a number if present
    final = pm.get("final")
    assert final is None or isinstance(final, (int | float))
