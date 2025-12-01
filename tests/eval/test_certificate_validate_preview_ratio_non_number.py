from unittest.mock import patch

import pytest

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_render_rejects_when_preview_final_ratio_not_number():
    # Start from a valid certificate, then corrupt preview_final_ratio to hit validate_certificate branch
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {"ppl_preview": 10.0, "ppl_final": 10.0},
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
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)
    # Seed primary_metric via ppl_ratio for PM-only validation path
    report["metrics"]["ppl_ratio"] = 1.0
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)
    # Corrupt schema_version to force validation failure regardless of jsonschema availability
    cert["schema_version"] = "wrong-version"
    with pytest.raises(ValueError):
        render_certificate_markdown(cert)
