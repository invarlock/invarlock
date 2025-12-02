from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_invariants_markdown_detail_pairs_rendered():
    # Minimal valid report/baseline
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

    # Inject invariants with a detailed failure dict to flip formatting branch
    cert["invariants"] = {
        "status": "fail",
        "summary": {"fatal_violations": 0, "warning_violations": 1},
        "failures": [
            {
                "check": "weight_norm",
                "type": "violation",
                "severity": "warning",
                "detail": {"layer": 3, "norm": 2.3, "note": "high"},
            }
        ],
    }

    md = render_certificate_markdown(cert)
    # Expect key=value pairs in parentheses
    assert "Invariant Notes" in md
    assert "layer=" in md and "norm=" in md and "note=" in md
