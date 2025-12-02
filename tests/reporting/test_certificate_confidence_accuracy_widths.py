from __future__ import annotations

from invarlock.reporting import certificate as C


def test_confidence_label_accuracy_threshold_override():
    # High confidence when width <= thr (pp-based for accuracy)
    cert = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {
            "kind": "accuracy",
            "display_ci": [0.700, 0.709],
            "unstable": False,
        },
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 1.0}},
    }
    out = C._compute_confidence_label(cert)
    assert out["basis"] == "accuracy"
    assert out["label"] in {"High", "Medium", "Low"}
