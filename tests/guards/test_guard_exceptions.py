from __future__ import annotations

import pytest

from invarlock.core.exceptions import GuardError, PolicyViolationError, ValidationError
from invarlock.guards.policies import (
    create_custom_rmt_policy,
    create_custom_spectral_policy,
    get_spectral_policy,
    get_validation_gate,
)


def test_policy_lookup_raises_guard_error_with_code() -> None:
    with pytest.raises(GuardError) as ei:
        get_spectral_policy("does_not_exist")
    assert getattr(ei.value, "code", None) == "E502"


def test_parameter_validation_raises_validation_error_with_code() -> None:
    with pytest.raises(ValidationError) as ei:
        create_custom_spectral_policy(sigma_quantile=2.0)  # invalid range
    assert getattr(ei.value, "code", None) == "E501"
    with pytest.raises(ValidationError) as ei2:
        create_custom_rmt_policy(margin=0.5)  # invalid margin
    assert getattr(ei2.value, "code", None) == "E501"


def test_enforce_validation_gate_raises_policy_violation() -> None:
    from invarlock.guards.policies import enforce_validation_gate

    gate = get_validation_gate("standard")
    # Violates max_capping_rate and max_ppl_degradation
    metrics = {
        "caps_applied": 8,
        "total_layers": 10,
        "primary_metric_ratio": 1.2,  # 20% degradation (ppl-like)
        "branch_balance_ok": False,
    }

    with pytest.raises(PolicyViolationError) as ei:
        enforce_validation_gate(metrics, gate)
    err = ei.value
    assert getattr(err, "code", None) == "E503"
    det = getattr(err, "details", {}) or {}
    assert isinstance(det.get("violations"), list) and det["violations"], det
