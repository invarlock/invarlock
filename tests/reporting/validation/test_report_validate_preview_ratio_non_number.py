from unittest.mock import patch

from invarlock.reporting.render import render_report_markdown

from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_schema import validate_report


def test_schema_rejects_wrong_version_before_render():
    # Start from a valid evaluation report, then corrupt schema_version.
    # Rendering is intentionally formatting-only; schema rejection belongs to report_schema.
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
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)
    # Seed primary_metric via ppl_ratio for PM-only validation path
    report["metrics"]["ppl_ratio"] = 1.0
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)
    # Corrupt schema_version to force validation failure regardless of jsonschema availability.
    cert["schema_version"] = "wrong-version"
    assert validate_report(cert) is False
    assert isinstance(render_report_markdown(cert), str)
