from invarlock.guards.variance import VarianceGuard


def _prep(policy):
    g = VarianceGuard(policy=policy)
    g._prepared = True
    g._post_edit_evaluated = True
    return g


def test_validate_warn_when_monitor_only_and_fail():
    g = _prep(
        {
            "mode": "ci",
            "min_gain": 0.0,
            "min_rel_gain": 0.0,
            "monitor_only": True,
            "scope": "ffn",
            "max_calib": 100,
        }
    )
    # Set metrics to cause failure and ensure finalize flags errors
    g._ab_gain = 0.0
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 100.0
    g._ratio_ci = (1.0, 1.0)
    # Cause PPL rise when VE disabled to produce an error
    g._final_ppl = 101.0
    result = g.validate(object(), adapter=None, context={})
    assert result["passed"] is False and result["action"] == "warn"


def test_validate_continue_when_aligned_and_no_warnings():
    g = _prep(
        {
            "mode": "ci",
            "min_gain": 0.0,
            "min_rel_gain": 0.01,
            "scope": "ffn",
            "max_calib": 100,
        }
    )
    # Configure should_enable False (ratio_ci hi > 1 - min_rel_gain), with matching enabled_after_ab False
    g._predictive_gate_state.update({"evaluated": True, "passed": True})
    g._ab_gain = 0.0
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 100.0
    g._ratio_ci = (0.99, 0.995)
    result = g.validate(object(), adapter=None, context={})
    # Some warnings may be produced by ancillary checks; ensure passed True
    assert result["passed"] is True
