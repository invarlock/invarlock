import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_finalize_happy_path_pass(monkeypatch):
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
    g._target_modules = {"transformer.h.0.mlp.c_proj": nn.Linear(2, 2, bias=False)}
    g._ppl_no_ve = 50.0
    g._ppl_with_ve = 49.0  # improvement 1.0 > absolute floor
    g._ab_gain = 0.02  # above tie breaker
    g._ratio_ci = (0.9, 0.95)
    g._calibration_stats = {"status": "complete"}
    g._predictive_gate_state = {}

    monkeypatch.setattr(g, "_evaluate_ab_gate", lambda: (True, "criteria_met"))

    def do_enable(_m):
        g._enabled = True
        return True

    monkeypatch.setattr(g, "enable", do_enable)
    monkeypatch.setattr(g, "disable", lambda m: None)
    monkeypatch.setattr(g, "_refresh_after_edit_metrics", lambda m: None)

    res = g.finalize(nn.Linear(2, 2))
    assert res.get("passed") is True and not res.get("errors")
