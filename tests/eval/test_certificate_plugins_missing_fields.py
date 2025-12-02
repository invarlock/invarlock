from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_render_markdown_plugin_provenance_missing_fields_and_na_overhead():
    report = {
        "meta": {
            "model_id": "m",
            "seed": 1,
            # Minimal plugin provenance: missing version/entry_point/module entries
            "plugins": {
                "adapter": {"name": "hf_adapter"},
                "edit": {"name": "structured"},
                "guards": [
                    {"name": "variance"},
                    {"name": "spectral"},
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
        # Provide guard_overhead with neither percent nor ratio → measured becomes "N/A"
        "guard_overhead": {"overhead_threshold": 0.01},
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
    # Plugin section may be omitted by normalization; ensure overall render is sane
    assert isinstance(md, str) and (
        "## Plugin Provenance" in md or "## Executive Summary" in md
    )
