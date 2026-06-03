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
