from invarlock.guards.variance import VarianceGuard


def test_set_ab_results_invalid_and_manual_override():
    g = VarianceGuard()
    # Invalid PPL values path
    g.set_ab_results(
        ppl_no_ve=0.0,
        ppl_with_ve=10.0,
        windows_used=None,
        seed_used=None,
        ratio_ci=None,
    )
    assert g._ab_gain == 0.0

    # Manual override when ratio_ci upper < 1.0
    g.set_ab_results(
        ppl_no_ve=100.0,
        ppl_with_ve=95.0,
        windows_used=3,
        seed_used=7,
        ratio_ci=(0.95, 0.98),
    )
    pg = g._predictive_gate_state
    assert pg.get("passed") is True and pg.get("reason") == "manual_override"
