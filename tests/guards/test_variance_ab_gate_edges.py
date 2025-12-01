from invarlock.guards.variance import VarianceGuard


def _fresh_guard():
    # Balanced defaults provide min_gain/min_rel_gain small enough for simple checks
    return VarianceGuard()


def test_ab_gate_missing_ratio_ci():
    g = _fresh_guard()
    g.set_ab_results(100.0, 90.0, ratio_ci=None)
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason == "missing_ratio_ci"


def test_ab_gate_ci_interval_too_high():
    g = _fresh_guard()
    # Valid PPLs, but CI upper bound above required_hi threshold
    g.set_ab_results(100.0, 90.0, ratio_ci=(0.9, 1.01))
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("ci_interval_too_high")


def test_ab_gate_predictive_gate_failed_overrides():
    g = _fresh_guard()
    g.set_ab_results(100.0, 90.0, ratio_ci=(0.9, 0.95))
    # Force predictive gate evaluated but failed
    g._predictive_gate_state = {
        "evaluated": True,
        "passed": False,
        "reason": "ci_contains_zero",
    }
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("predictive_gate_failed")


def test_ab_gate_below_absolute_floor():
    # Force min_rel_gain to 0 so absolute floor check triggers next
    g = VarianceGuard(policy={"min_rel_gain": 0.0})
    # Improvement below default ABSOLUTE_FLOOR (0.05)
    g.set_ab_results(100.00, 99.98, ratio_ci=(0.8, 0.9))
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("below_absolute_floor")
