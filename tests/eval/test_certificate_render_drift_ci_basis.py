from unittest.mock import patch

from invarlock.reporting.report_builder import make_report
from invarlock.reporting.render import render_report_markdown


def test_drift_basis_includes_ci_informational_when_ci_present():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "ppl_drift_ci": (0.98, 1.02),
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
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with patch("invarlock.reporting.report_builder.validate_run_report", return_value=True):
        cert = make_report(report, baseline)

    md = render_report_markdown(cert)
    # PM-first: Basis cell shows 'point' when CI present
    assert "| point |" in md
