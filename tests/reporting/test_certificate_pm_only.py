from __future__ import annotations

from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_certificate_has_no_ppl_block_pm_only():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            # Minimal ppl-like inputs; PM fallback should be populated automatically
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "ppl_ratio": 1.0,
        },
        "data": {
            "dataset": "dummy",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [],
        "edit": {
            "name": "structured",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    baseline = {"run_id": "b", "model_id": "m", "ppl_final": 10.0}

    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)

    # PM-only: no 'ppl' key should be present in the certificate
    assert "ppl" not in cert
    # Primary metric should exist
    assert isinstance(cert.get("primary_metric"), dict) and cert["primary_metric"]
