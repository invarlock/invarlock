import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_finalize_absolute_floor_error_when_enabled_but_improvement_small(monkeypatch):
    g = VarianceGuard(
        policy={"min_gain": 0.0, "seed": 1, "scope": "both", "max_calib": 0}
    )

    # Minimal state
    g._enabled = False
    g._ppl_no_ve = 50.00
    g._ppl_with_ve = 49.53  # improvement 0.47 < ABSOLUTE_FLOOR (0.5)
    g._ab_gain = 0.01
    g._ratio_ci = (0.98, 1.01)
    g._scales = {}
    g._target_modules = {}
    g._calibration_stats = {}
    g._predictive_gate_state = {"passed": True, "evaluated": True}
    g._prepared = True
    # Ensure absolute floor is strict enough to fail (default is 0.05)
    g.ABSOLUTE_FLOOR = 0.5

    # Force A/B gate approval and successful enable
    monkeypatch.setattr(g, "_evaluate_ab_gate", lambda: (True, "criteria_met"))

    def do_enable(_m):
        g._enabled = True
        return True

    monkeypatch.setattr(g, "enable", do_enable)
    monkeypatch.setattr(g, "disable", lambda m: None)
    monkeypatch.setattr(g, "_refresh_after_edit_metrics", lambda m: None)

    res = g.finalize(nn.Linear(2, 2))
    # Expect absolute floor error message present
    errs = ", ".join(res.get("errors", []))
    assert "absolute floor" in errs
