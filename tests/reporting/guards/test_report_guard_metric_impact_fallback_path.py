from copy import deepcopy

import invarlock.reporting.report_normalization as report_normalization
from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_guard_metric_impact_direct_values_and_unavailable_measurement_path(
    monkeypatch,
):
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
                "ratio_vs_baseline": float("nan"),
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
        # Inject a guard_metric_impact block with bare/guarded values but invalid (bare_value <= 0) → ratio unavailable path
        "guard_metric_impact": {
            "bare_value": 0.0,
            "guarded_value": 10.0,
            "degradation_limit": 0.01,
        },
    }
    baseline = deepcopy(report)
    baseline["run_id"] = "b"
    baseline["edit"]["name"] = "noop"
    baseline["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "final": 10.0,
        "preview": 10.0,
    }
    monkeypatch.setattr(report_normalization, "validate_report", lambda _: True)
    cert = make_report(report, baseline)
    # Guard metric impact section may be omitted; renderer should handle it gracefully
    oh = cert.get("guard_metric_impact", {})
    assert isinstance(oh, dict)
    _ = render_report_markdown(cert)
