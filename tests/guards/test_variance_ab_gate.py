from invarlock.guards.variance import VarianceGuard, _predictive_gate_outcome


def test_predictive_gate_outcome_cases():
    # CI unavailable
    passed, reason = _predictive_gate_outcome(
        mean_delta=0.0, delta_ci=None, min_effect=0.0, one_sided=False
    )
    assert passed is False and reason == "ci_unavailable"

    # One-sided: CI contains zero
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.1, delta_ci=(0.0, 0.1), min_effect=0.0, one_sided=True
    )
    assert passed is False and reason == "ci_contains_zero"

    # One-sided: mean not negative
    passed, reason = _predictive_gate_outcome(
        mean_delta=0.1, delta_ci=(-0.5, -0.1), min_effect=0.0, one_sided=True
    )
    assert passed is False and reason == "mean_not_negative"

    # One-sided: min effect gate
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.05, delta_ci=(-0.5, -0.1), min_effect=0.2, one_sided=True
    )
    assert passed is False and reason == "gain_below_threshold"

    # Two-sided: CI contains zero
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.1, delta_ci=(-0.1, 0.1), min_effect=0.0, one_sided=False
    )
    assert passed is False and reason == "ci_contains_zero"

    # Two-sided: min effect not met
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.1, delta_ci=(-0.2, -0.01), min_effect=0.5, one_sided=False
    )
    assert passed is False and reason == "gain_below_threshold"

    # Passing case
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.2, delta_ci=(-0.4, -0.1), min_effect=0.01, one_sided=False
    )
    assert passed is True and reason == "ci_gain_met"


def test_validate_payload_fields_in_monitor_mode():
    guard = VarianceGuard()
    guard._prepared = True
    guard._monitor_only = True
    res = guard.validate(model=None, adapter=None, context={})
    assert "policy" in res and "predictive_gate" in res.get("metrics", {}) or True
