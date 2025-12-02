from invarlock.guards.variance import _predictive_gate_outcome


def test_predictive_gate_outcome_one_sided_reasons():
    # CI contains zero -> fail
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.001, delta_ci=(0.0, 0.01), min_effect=0.0, one_sided=True
    )
    assert passed is False and reason == "ci_contains_zero"

    # Mean not negative -> fail
    passed, reason = _predictive_gate_outcome(
        mean_delta=0.01, delta_ci=(-0.02, -0.001), min_effect=0.0, one_sided=True
    )
    assert passed is False and reason == "mean_not_negative"

    # Gain below threshold -> fail
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.001, delta_ci=(-0.002, -0.0001), min_effect=0.01, one_sided=True
    )
    assert passed is False and reason == "gain_below_threshold"


def test_predictive_gate_outcome_two_sided_reasons():
    # Upper >= 0 -> fail
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.01, delta_ci=(-0.1, 0.0), min_effect=0.0, one_sided=False
    )
    assert passed is False and reason == "ci_contains_zero"

    # Gain below threshold with two-sided
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.01, delta_ci=(-0.02, -0.001), min_effect=0.05, one_sided=False
    )
    assert passed is False and reason == "gain_below_threshold"
