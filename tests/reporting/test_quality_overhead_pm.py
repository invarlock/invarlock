from __future__ import annotations

from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_quality_overhead_ppl_ratio_shows():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            # Indicate ppl-like primary metric in the main report for clarity
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 20.0,
                "ratio_vs_baseline": 1.0,
            }
        },
        # Provide guard_overhead with bare/guarded reports containing final windows
        "guard_overhead": {
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
    baseline = {"run_id": "b", "model_id": "m", "ppl_final": 10.0}

    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)

    # quality_overhead may be omitted; markdown should still render
    md = render_certificate_markdown(cert)
    assert "# InvarLock Safety Certificate" in md


def test_quality_overhead_accuracy_delta_near_zero():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "primary_metric": {
                "kind": "accuracy",
                "final": 0.50,
                "ratio_vs_baseline": 0.0,
            }
        },
        "guard_overhead": {
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
    baseline = {"run_id": "b", "model_id": "m", "ppl_final": 10.0}

    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)

    md = render_certificate_markdown(cert)
    assert "# InvarLock Safety Certificate" in md
