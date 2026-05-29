from __future__ import annotations

from invarlock.reporting.report_enrichment import compute_confidence_label


def test_confidence_label_accuracy_basis():
    cert = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "accuracy", "display_ci": [0.80, 0.82]},
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 3.0}},
    }
    out = compute_confidence_label(cert)
    assert out["basis"] == "accuracy"
    assert out["label"] == "High"
