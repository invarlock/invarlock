from invarlock.guards.variance import VarianceGuard


def _guard_with_defaults(**overrides) -> VarianceGuard:
    policy = {
        "min_gain": 0.0,
        "scope": "both",
        "max_calib": 0,
        "mode": "ci",
        "min_rel_gain": 0.001,
    }
    policy.update(overrides)
    return VarianceGuard(policy=policy)


def test_evaluate_ab_gate_no_results_and_invalid_ppl():
    g = _guard_with_defaults()
    # No _ab_gain set
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason == "no_ab_results"

    # Set ab, but invalid ppl values
    g._ab_gain = 0.01
    g._ppl_no_ve = 0.0
    g._ppl_with_ve = 49.0
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason == "invalid_ppl_values"


def test_evaluate_ab_gate_below_min_rel_and_min_effect_lognll():
    g = _guard_with_defaults(min_rel_gain=0.02, min_effect_lognll=0.5)
    g._ab_gain = 0.01  # below min_rel_gain
    g._ppl_no_ve = 50.0
    g._ppl_with_ve = 49.0
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("below_min_rel_gain")

    # Satisfy rel gain but fail min_effect_lognll
    g = _guard_with_defaults(min_rel_gain=0.0, min_effect_lognll=0.1)
    g._ab_gain = 0.05
    g._ppl_no_ve = 50.0
    g._ppl_with_ve = 49.9  # log gain ~ ln(50/49.9) ~ 0.002
    g._ratio_ci = (0.98, 1.02)
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("below_min_effect_lognll")


def test_evaluate_ab_gate_missing_ratio_and_ci_too_high():
    g = _guard_with_defaults(min_rel_gain=0.0)
    g._ab_gain = 0.1
    g._ppl_no_ve = 50.0
    g._ppl_with_ve = 45.0
    # missing ratio_ci
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason == "missing_ratio_ci"

    # Provide ci with high upper bound, forcing ci_interval_too_high
    g._ratio_ci = (0.8, 1.05)
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("ci_interval_too_high")
