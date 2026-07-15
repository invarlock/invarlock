from copy import deepcopy
from unittest.mock import patch

from invarlock.reporting.rendering.markdown import (
    render_report_markdown,
)
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_render_spectral_multiple_testing_yaml_block():
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            }
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
    baseline = deepcopy(report)
    baseline["run_id"] = "b"
    baseline["edit"]["name"] = "noop"
    baseline["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
    }
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)

    # Inject multiple_testing info into spectral and render
    cert.setdefault("spectral", {})["multiple_testing"] = {
        "method": "bh",
        "alpha": 0.05,
        "m": 4,
    }
    md = render_report_markdown(cert)
    assert "Multiple Testing" in md and "method: bh" in md
