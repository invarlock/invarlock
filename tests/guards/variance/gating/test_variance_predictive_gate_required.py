from invarlock.guards.variance import VarianceGuard


def test_evaluate_ab_gate_rejects_unevaluated_predictive_gate() -> None:
    guard = VarianceGuard(policy={"mode": "ci", "min_gain": 0.0, "min_rel_gain": 0.0})
    guard._ab_gain = 0.10
    guard._ppl_no_ve = 100.0
    guard._ppl_with_ve = 90.0
    guard._ratio_ci = (0.80, 0.95)
    guard._predictive_gate_state = {}

    ok, reason = guard._evaluate_ab_gate()

    assert ok is False
    assert reason == "predictive_gate_failed (predictive_gate_failed)"
    assert guard._predictive_gate_state == {}
