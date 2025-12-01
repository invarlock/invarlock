from __future__ import annotations

from invarlock.reporting.certificate import _compute_thresholds_payload


def test_thresholds_payload_includes_accuracy_params():
    payload = _compute_thresholds_payload("balanced", resolved_policy={})
    assert "accuracy" in payload
    acc = payload["accuracy"]
    # Ensure keys are mapped and numeric
    assert set(acc).issuperset(
        {"delta_min_pp", "min_examples", "min_examples_fraction", "hysteresis_delta_pp"}
    )
    assert isinstance(acc["delta_min_pp"], float)
    assert isinstance(acc["min_examples"], int)
