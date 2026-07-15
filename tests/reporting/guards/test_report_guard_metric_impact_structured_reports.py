from copy import deepcopy
from unittest.mock import patch

from invarlock.reporting.rendering.markdown import (
    render_report_markdown,
)
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_guard_metric_impact_structured_reports_path():
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
        "plugins": {"adapter": {}, "edit": {}, "guards": []},
        # Structured reports path
        "guard_metric_impact": {
            "degradation_limit": 0.02,
            "bare_report": {"metrics": {"ppl_final": 10.0}},
            "guarded_report": {"metrics": {"ppl_final": 10.09}},
        },
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
    oh = cert.get("guard_metric_impact", {})
    # Guard metric impact may be omitted; renderer should still succeed
    assert isinstance(oh, dict)
    _ = render_report_markdown(cert)
