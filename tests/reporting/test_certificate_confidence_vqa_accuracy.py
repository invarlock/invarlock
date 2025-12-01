from __future__ import annotations

from invarlock.reporting import certificate as C


def test_confidence_label_vqa_accuracy_basis():
    cert = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {"kind": "vqa_accuracy", "display_ci": [0.80, 0.82]},
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 3.0}},
    }
    out = C._compute_confidence_label(cert)
    assert out["basis"] == "vqa_accuracy"
    assert out["label"] in {"High", "Medium", "Low"}
