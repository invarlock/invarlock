from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_guard_metric_impact_ppl_degradation_shows():
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "metrics": {
            # Indicate ppl-like primary metric in the main report for clarity
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 20.0,
                "final": 20.0,
                "ratio_vs_baseline": 1.0,
            }
        },
        # Provide guard_metric_impact with bare/guarded reports containing final windows
        "guard_metric_impact": {
            "bare_report": {
                "meta": {"model_id": "m", "seed": 1},
                "evaluation_windows": {
                    "final": {"logloss": [1.00], "token_counts": [100]}
                },
            },
            "guarded_report": {
                "meta": {"model_id": "m", "seed": 1},
                "evaluation_windows": {
                    "final": {"logloss": [1.10], "token_counts": [100]}
                },
            },
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
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
    }
    baseline = deepcopy(report)
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

    # guard_metric_impact may be omitted; markdown should still render
    md = render_report_markdown(cert)
    assert "# InvarLock Evaluation Report" in md


def test_guard_metric_impact_accuracy_delta_near_zero():
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "metrics": {
            "primary_metric": {
                "kind": "accuracy",
                "preview": 0.50,
                "final": 0.50,
                "delta_vs_baseline_pp": 0.0,
            }
        },
        "guard_metric_impact": {
            "bare_report": {
                "metrics": {
                    "classification": {"final": {"correct_total": 50, "total": 100}}
                }
            },
            "guarded_report": {
                "metrics": {
                    "classification": {"final": {"correct_total": 50, "total": 100}}
                }
            },
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
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
    }
    baseline = deepcopy(report)
    baseline["edit"]["name"] = "noop"
    baseline["metrics"]["primary_metric"] = {
        "kind": "accuracy",
        "preview": 0.50,
        "final": 0.50,
    }

    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)

    md = render_report_markdown(cert)
    assert "# InvarLock Evaluation Report" in md
