import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_finalize_enabled_but_below_tie_breaker_deadband_adds_error(monkeypatch):
    # Use balanced defaults: min_gain 0.0, tie_breaker_deadband ~0.001
    g = VarianceGuard(
        policy={
            "min_gain": 0.0,
            "scope": "both",
            "max_calib": 0,
            "tie_breaker_deadband": 0.001,
        }
    )
    g._prepared = True
    g._enabled = False
    g._ppl_no_ve = 50.0
    g._ppl_with_ve = 49.4  # meets absolute floor easily
    g._ab_gain = 0.0005  # less than min_gain + tie_breaker_deadband
    g._ratio_ci = (0.98, 1.01)
    g._scales = {}
    g._target_modules = {"transformer.h.0.mlp.c_proj": nn.Linear(2, 2, bias=False)}
    g._calibration_stats = {}
    g._predictive_gate_state = {"passed": True, "evaluated": True}

    # Gate approves; enable succeeds -> enabled_after_ab True
    monkeypatch.setattr(g, "_evaluate_ab_gate", lambda: (True, "criteria_met"))

    def do_enable(_m):
        g._enabled = True
        return True

    monkeypatch.setattr(g, "enable", do_enable)
    monkeypatch.setattr(g, "disable", lambda m: None)
    monkeypatch.setattr(g, "_refresh_after_edit_metrics", lambda m: None)

    res = g.finalize(nn.Linear(2, 2))
    errs = ", ".join(res.get("errors", []))
    assert "tie-breaker" in errs or "deadband" in errs
