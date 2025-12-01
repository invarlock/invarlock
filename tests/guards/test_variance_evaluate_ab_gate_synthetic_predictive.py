from invarlock.guards.variance import VarianceGuard


def test_evaluate_ab_gate_sets_synthetic_predictive_state():
    g = VarianceGuard(
        policy={"min_gain": 0.0, "scope": "both", "max_calib": 0, "mode": "ci"}
    )
    g._ab_gain = 0.5
    g._ppl_no_ve = 50.0
    g._ppl_with_ve = 25.0
    # Having ratio_ci present and gate_state empty triggers synthetic predictive evaluated/pass
    g._ratio_ci = (0.4, 0.8)
    ok, reason = g._evaluate_ab_gate()
    assert g._predictive_gate_state.get("evaluated") is True
    assert ok in {True, False} and isinstance(reason, str)
