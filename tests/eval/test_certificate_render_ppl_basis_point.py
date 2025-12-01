from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_render_markdown_uses_point_basis_when_no_ratio_ci():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {"ppl_preview": 10.0, "ppl_final": 10.0, "ppl_ratio": 1.0},
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
    # Lower the auto target ratio to exercise the ratio_limit=min(...) path
    cert.setdefault("auto", {})["tier"] = "balanced"
    cert["auto"]["target_pm_ratio"] = 1.05

    md = render_certificate_markdown(cert)
    # Quality gates section present; row headings may vary across versions
    assert "Quality Gates" in md
    assert "| point |" in md
