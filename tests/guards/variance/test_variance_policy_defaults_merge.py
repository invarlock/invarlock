from invarlock.guards.variance import VarianceGuard


def test_variance_guard_merges_defaults_for_partial_policy():
    g = VarianceGuard(policy={"calibration": {"windows": 7}})

    assert g._policy.get("scope") in {"ffn", "attn", "both"}
    assert isinstance(g._policy.get("min_gain"), (int, float))

    calibration = g._policy.get("calibration")
    assert isinstance(calibration, dict)
    assert calibration.get("windows") == 7
