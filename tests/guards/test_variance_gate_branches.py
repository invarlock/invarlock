from invarlock.guards.variance import VarianceGuard


def make_guard(policy):
    g = VarianceGuard(policy=policy)
    # Pre-set post-edit metrics to avoid heavy computation
    g._prepared = True
    g._post_edit_evaluated = True
    g._predictive_gate_state.update(
        {"evaluated": True, "passed": True, "reason": "ci_gain_met"}
    )
    return g


def test_ab_gate_delta_mode_tie_breaker_and_thresholds():
    # tie-breaker deadband disables on marginal gains
    g = make_guard(
        {
            "mode": "delta",
            "min_gain": 0.01,
            "min_rel_gain": 0.005,
            "tie_breaker_deadband": 0.005,
        }
    )
    g._ab_gain = 0.012  # below min_gain + tie_breaker (0.015)
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 99.0
    g._ratio_ci = (0.98, 0.99)
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and "below_threshold_with_deadband" in reason

    # Above thresholds
    g._ab_gain = 0.02
    ok, reason = g._evaluate_ab_gate()
    assert ok is True


def test_ab_gate_ci_mode_bounds_and_min_effect():
    g = make_guard(
        {
            "mode": "ci",
            "min_gain": 0.0,
            "min_rel_gain": 0.01,
            "min_effect_lognll": 0.001,
        }
    )
    g._ab_gain = 0.02
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 98.0
    # CI hi must be <= 1 - min_rel_gain AND exp(-min_effect_lognll)
    g._ratio_ci = (0.90, 0.995)  # hi slightly above 0.99 requirement
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and "ci_interval_too_high" in reason

    g._ratio_ci = (0.90, 0.98)
    ok, reason = g._evaluate_ab_gate()
    assert ok is True


def test_ab_gate_invalid_values_and_monitor_only():
    g = make_guard(
        {"mode": "ci", "min_gain": 0.0, "min_rel_gain": 0.0, "monitor_only": True}
    )
    g._ab_gain = 0.0
    # invalid ppl values
    g._ppl_no_ve = 0.0
    g._ppl_with_ve = 0.0
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and "invalid_ppl_values" in reason


def test_predictive_gate_disabled_and_below_absolute_floor():
    # predictive gate disabled allows pass if thresholds met
    g = make_guard(
        {"mode": "ci", "min_gain": 0.0, "min_rel_gain": 0.0, "predictive_gate": False}
    )
    g._ab_gain = 0.05
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 95.0
    g._ratio_ci = (0.90, 0.95)
    ok, reason = g._evaluate_ab_gate()
    assert ok is True

    # Below absolute floor prevents enabling
    g2 = make_guard({"mode": "delta", "min_gain": 0.0, "min_rel_gain": 0.0})
    g2._ab_gain = 0.1
    g2._ppl_no_ve = 100.0
    g2._ppl_with_ve = 99.96  # improvement 0.04 < 0.05
    g2._ratio_ci = (0.98, 0.99)
    ok2, reason2 = g2._evaluate_ab_gate()
    assert ok2 is False and "below_absolute_floor" in reason2
