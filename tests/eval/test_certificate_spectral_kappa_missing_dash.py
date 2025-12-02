from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_spectral_family_caps_kappa_missing_renders_dash():
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

    # Inject spectral data: caps_by_family present, but family_caps lacks numeric kappa
    cert["spectral"] = {
        "caps_applied": 1,
        "max_caps": 5,
        "summary": {"caps_exceeded": False},
        "caps_applied_by_family": {"ffn": 3},
        "family_caps": {"ffn": {"kappa": float("nan")}},
    }
    md = render_certificate_markdown(cert)
    # Expect dash in κ column
    assert "| Family | κ | Violations |" in md
    assert "| ffn | - | 3 |" in md
