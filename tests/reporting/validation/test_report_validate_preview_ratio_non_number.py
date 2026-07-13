from unittest.mock import patch

from invarlock.reporting.rendering.markdown import render_report_markdown
from invarlock.reporting.report_schema import validate_report
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_schema_rejects_wrong_version_before_render():
    # Start from a valid evaluation report, then corrupt schema_version.
    # Rendering is intentionally formatting-only; schema rejection belongs to report_schema.
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev", "assurance": {"mode": "off"}},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
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
    baseline = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev", "assurance": {"mode": "off"}},
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
        "artifacts": {"events_path": "", "logs_path": ""},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)
    # Corrupt schema_version to force validation failure regardless of jsonschema availability.
    cert["schema_version"] = "wrong-version"
    assert validate_report(cert) is False
    assert isinstance(render_report_markdown(cert), str)
