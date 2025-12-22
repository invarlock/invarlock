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

    # Regression outside +min_effect band -> fail with explicit reason
    passed, reason = _predictive_gate_outcome(
        mean_delta=0.02, delta_ci=(0.015, 0.03), min_effect=0.01, one_sided=False
    )
    assert passed is False and reason == "regression_detected"


def test_predictive_gate_outcome_one_sided_mean_above_threshold():
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.005, delta_ci=(-0.05, -0.02), min_effect=0.01, one_sided=True
    )
    assert passed is False and reason == "gain_below_threshold"


def test_predictive_gate_outcome_two_sided_positive_low_gain():
    passed, reason = _predictive_gate_outcome(
        mean_delta=0.005, delta_ci=(0.005, 0.02), min_effect=0.01, one_sided=False
    )
    assert passed is False and reason == "mean_not_negative"


def test_predictive_gate_outcome_two_sided_mean_nonnegative_path():
    passed, reason = _predictive_gate_outcome(
        mean_delta=0.01, delta_ci=(-0.2, -0.02), min_effect=0.0, one_sided=False
    )
    assert passed is False and reason == "mean_not_negative"


def test_predictive_gate_outcome_two_sided_mean_below_min_effect():
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.01, delta_ci=(-0.2, -0.06), min_effect=0.05, one_sided=False
    )
    assert passed is False and reason == "gain_below_threshold"
