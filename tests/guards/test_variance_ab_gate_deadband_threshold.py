from invarlock.guards.variance import VarianceGuard


def test_ab_gate_below_threshold_with_deadband_reason():
    g = VarianceGuard(policy={"mode": "ci", "min_gain": 0.05, "min_rel_gain": 0.0})
    g._ab_gain = 0.052  # below 0.05 + deadband(0.005) = 0.055
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 94.0  # absolute improvement > 0.05
    g._ratio_ci = (0.7, 0.8)  # acceptable upper bound
    # Predictive gate already passed
    g._predictive_gate_state = {
        "evaluated": True,
        "passed": True,
        "reason": "ci_gain_met",
    }
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("below_threshold_with_deadband")
