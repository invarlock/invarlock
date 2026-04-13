from __future__ import annotations

import pytest

from invarlock.reporting.report_confidence import compute_confidence_label


def test_confidence_label_accuracy_threshold_override():
    # High confidence when width <= threshold measured in percentage points.
    cert = {
        "validation": {"primary_metric_acceptable": True},
        "primary_metric": {
            "kind": "accuracy",
            "display_ci": [70.0, 70.9],
            "unstable": False,
        },
        "resolved_policy": {"confidence": {"accuracy_delta_pp_width_max": 1.0}},
    }
    out = compute_confidence_label(cert)
    assert out["basis"] == "accuracy"
    assert out["width"] == pytest.approx(0.9)
    assert out["label"] == "High"
