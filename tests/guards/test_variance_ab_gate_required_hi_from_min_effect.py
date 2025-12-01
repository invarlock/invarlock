from invarlock.guards.variance import VarianceGuard


def test_ab_gate_ci_too_high_uses_min_effect_lognll_threshold():
    g = VarianceGuard(
        policy={
            "mode": "ci",
            "min_gain": 0.0,
            "min_rel_gain": 0.0,
            "min_effect_lognll": 0.2,
        }
    )
    g._ab_gain = 0.25
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 80.0  # log gain ~ 0.223 > 0.2 → passes min_effect_lognll gate
    # required_hi = min(1.0, exp(-0.2)) ~ 0.81873; set hi slightly above to force ci_interval_too_high
    g._ratio_ci = (0.7, 0.83)
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("ci_interval_too_high")
