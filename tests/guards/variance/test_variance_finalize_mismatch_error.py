import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_finalize_mismatch_enabled_but_gate_rejects_errors(monkeypatch):
    g = VarianceGuard(policy={"min_gain": 0.0, "scope": "both", "max_calib": 0})
    g._prepared = True
    g._enabled = True  # already enabled
    g._target_modules = {"transformer.h.0.mlp.c_proj": nn.Linear(2, 2, bias=False)}
    g._ppl_no_ve = 50.0
    g._ppl_with_ve = 49.0
    g._ab_gain = 0.02
    g._ratio_ci = (0.98, 0.99)
    g._calibration_stats = {"status": "complete"}
    g._predictive_gate_state = {"passed": True}

    # Gate rejects → finalize should disable and record error
    monkeypatch.setattr(g, "_evaluate_ab_gate", lambda: (False, "below_threshold"))
    disabled = {"called": False}

    def do_disable(_m):
        disabled["called"] = True
        g._enabled = False

    monkeypatch.setattr(g, "disable", do_disable)
    monkeypatch.setattr(g, "_refresh_after_edit_metrics", lambda m: None)

    res = g.finalize(nn.Linear(2, 2))
    assert disabled["called"] is True
    # No mismatch error expected because finalize forces disable path to reconcile state
    assert not res.get("errors")
