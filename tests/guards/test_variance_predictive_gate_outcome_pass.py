from invarlock.guards.variance import _predictive_gate_outcome


def test_predictive_gate_outcome_pass_one_sided():
    passed, reason = _predictive_gate_outcome(
        mean_delta=-0.1, delta_ci=(-0.2, -0.05), min_effect=0.0, one_sided=True
    )
    assert passed is True and reason == "ci_gain_met"
