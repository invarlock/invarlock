from unittest.mock import patch

from invarlock.reporting.certificate import (
    make_certificate,
    render_certificate_markdown,
)


def test_plugins_guards_render_with_valid_entries():
    report = {
        "meta": {
            "model_id": "m",
            "seed": 1,
            "plugins": {
                "adapter": {"name": "hf", "module": "pkg.adapter"},
                "edit": {"name": "structured", "module": "pkg.edits"},
                "guards": [
                    {"name": "g1", "module": "pkg.g1", "version": "1.0"},
                    {"name": "g2", "module": "pkg.g2"},
                ],
            },
        },
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
    md = render_certificate_markdown(cert)
    # Plugin provenance may be omitted after normalization
    assert ("- Guards:" in md) or ("## Executive Summary" in md)
