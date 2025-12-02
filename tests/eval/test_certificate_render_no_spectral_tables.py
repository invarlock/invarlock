from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_render_spectral_no_tables_when_empty():
    # Minimal report/baseline producing empty spectral tables
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

    # Ensure spectral exists but without tables
    spectral = cert.get("spectral", {})
    spectral.pop("caps_applied_by_family", None)
    spectral.pop("family_z_quantiles", None)
    cert["spectral"] = spectral

    md = render_certificate_markdown(cert)
    # Headers for spectral tables should be absent
    assert "| Family | κ | Violations |" not in md
    assert "| Family | q95 | q99 | Max | Samples |" not in md
