from copy import deepcopy
from unittest.mock import patch

from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_spectral_family_caps_kappa_missing_renders_dash():
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

    # Inject spectral data: caps_by_family present, but family_caps lacks numeric kappa
    cert["spectral"] = {
        "caps_applied": 1,
        "max_caps": 5,
        "summary": {"caps_exceeded": False},
        "caps_applied_by_family": {"ffn": 3},
        "family_caps": {"ffn": {"kappa": float("nan")}},
    }
    md = render_report_markdown(cert)
    # Expect dash in κ column
    assert "| Family | κ | q95 | Max |z| | Caps Applied |" in md
    assert "| ffn | - | - | - | 3 |" in md
