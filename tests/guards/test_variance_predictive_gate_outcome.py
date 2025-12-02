from invarlock.guards.variance import _predictive_gate_outcome


def test_predictive_gate_outcome_one_sided_pass():
    ok, reason = _predictive_gate_outcome(
        mean_delta=-0.1, delta_ci=(-0.2, -0.05), min_effect=0.01, one_sided=True
    )
    assert ok is True and reason == "ci_gain_met"


def test_predictive_gate_outcome_two_sided_zero_in_ci_fails():
    ok, reason = _predictive_gate_outcome(
        mean_delta=-0.1, delta_ci=(-0.2, 0.0), min_effect=0.0, one_sided=False
    )
    assert ok is False and reason == "ci_contains_zero"
