from unittest.mock import patch

from invarlock.reporting.render import render_report_markdown
from invarlock.reporting.report_make import make_report


def test_render_spectral_omits_policy_yaml_when_absent():
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

    # Ensure spectral has no policy section
    spectral = cert.get("spectral", {})
    spectral.pop("policy", None)
    cert["spectral"] = spectral

    md = render_report_markdown(cert)
    # The policy YAML header should not appear when absent
    assert "Family κ (policy):" not in md
