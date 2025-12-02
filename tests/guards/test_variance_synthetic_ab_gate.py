from invarlock.guards.variance import VarianceGuard


def test_evaluate_ab_gate_sets_synthetic_when_predictive_not_evaluated_but_ratio_ci_present():
    g = VarianceGuard(policy={"mode": "ci", "min_gain": 0.0, "min_rel_gain": 0.0})
    # Provide valid A/B results and ratio CI, but leave predictive gate state empty (not evaluated)
    g._ab_gain = 0.10
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 90.0
    g._ratio_ci = (0.80, 0.95)
    g._predictive_gate_state = {}

    ok, _ = g._evaluate_ab_gate()
    # Decision should proceed; importantly, predictive gate state should be marked evaluated with synthetic reason
    assert isinstance(ok, bool)
    state = g._predictive_gate_state
    assert state.get("evaluated") is True
    assert state.get("reason") == "synthetic_ab_gate"
