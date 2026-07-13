from unittest.mock import patch

from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_invariants_markdown_detail_pairs_rendered():
    # Minimal valid report/baseline
    report = {
        "meta": {
            "adapter": "hf",
            "model_id": "m",
            "seed": 1,
            "auto": {"tier": "balanced"},
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            }
        },
        "context": {"profile": "dev"},
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
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev"},
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {"name": "noop"},
        "guards": [],
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            }
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)

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

    md = render_report_markdown(cert)
    # Expect key=value pairs in parentheses
    assert "Invariant Notes" in md
    assert "layer=" in md and "norm=" in md and "note=" in md
