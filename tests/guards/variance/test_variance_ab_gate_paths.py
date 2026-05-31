from invarlock.guards.variance import VarianceGuard


def _base_guard():
    return VarianceGuard(
        policy={
            "mode": "ci",
            "min_gain": 0.05,
            "min_rel_gain": 0.01,
            "absolute_floor_ppl": 0.05,
            "predictive_gate": True,
        }
    )


def test_ab_gate_missing_ratio_ci():
    g = _base_guard()
    g._ab_gain = 0.2
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 90.0
    g._ratio_ci = None
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("missing_ratio_ci")


def test_ab_gate_invalid_ratio_ci():
    g = _base_guard()
    g._ab_gain = 0.2
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 90.0
    g._ratio_ci = (0.0, -1.0)
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("invalid_ratio_ci")


def test_ab_gate_below_min_rel_gain():
    g = _base_guard()
    g._ab_gain = 0.005  # below min_rel_gain=0.01
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 99.6
    g._ratio_ci = (0.9, 0.95)
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("below_min_rel_gain")


def test_ab_gate_predictive_gate_failed():
    g = _base_guard()
    g._ab_gain = 0.2
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 90.0
    g._ratio_ci = (0.8, 0.85)
    # Predictive gate explicitly failed
    g._predictive_gate_state = {
        "evaluated": True,
        "passed": False,
        "reason": "insufficient_coverage",
    }
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("predictive_gate_failed")
