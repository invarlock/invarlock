from invarlock.guards.variance import VarianceGuard


def test_ab_gate_below_deadband_with_valid_ci():
    g = VarianceGuard(
        policy={
            "mode": "ci",
            "min_gain": 0.1,
            "min_rel_gain": 0.0,
            "predictive_gate": False,
        }
    )
    g._ab_gain = 0.103  # less than 0.105 required (min_gain + default deadband 0.005)
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 90.0
    g._ratio_ci = (0.7, 0.9)  # valid interval well below 1.0
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("below_threshold_with_deadband")


def test_ab_gate_below_absolute_floor():
    g = VarianceGuard(
        policy={
            "mode": "ci",
            "min_gain": 0.0,
            "predictive_gate": False,
            "absolute_floor_ppl": 0.05,
        }
    )
    g._ab_gain = 0.2
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 99.97  # improvement = 0.03 < 0.05 floor
    g._ratio_ci = (0.7, 0.9)
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("below_absolute_floor")


def test_ab_gate_honors_custom_absolute_floor_from_initial_policy():
    g = VarianceGuard(
        policy={
            "mode": "ci",
            "min_gain": 0.0,
            "min_rel_gain": 0.0,
            "min_effect_lognll": 0.0,
            "predictive_gate": False,
            "absolute_floor_ppl": 10.0,
        }
    )
    g._ab_gain = 0.02
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 98.0
    g._ratio_ci = (0.97, 0.99)

    ok, reason = g._evaluate_ab_gate()

    assert g.ABSOLUTE_FLOOR == 10.0
    assert ok is False
    assert reason.startswith("below_absolute_floor")


def test_ab_gate_accepts_ratio_upper_equal_one_when_policy_thresholds_are_zero():
    g = VarianceGuard(
        policy={
            "mode": "ci",
            "min_gain": 0.0,
            "tie_breaker_deadband": 0.0,
            "min_rel_gain": 0.0,
            "min_effect_lognll": 0.0,
            "predictive_gate": False,
            "absolute_floor_ppl": 0.0,
        }
    )
    g._ab_gain = 0.01
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 99.0
    g._ratio_ci = (0.99, 1.0)

    ok, reason = g._evaluate_ab_gate()

    assert ok is True
    assert reason.startswith("criteria_met")
