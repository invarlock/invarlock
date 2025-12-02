import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_finalize_seed_mismatch_emits_warning(monkeypatch):
    g = VarianceGuard(
        policy={"min_gain": 0.0, "seed": 123, "scope": "both", "max_calib": 0}
    )
    g._prepared = True
    g._enabled = False
    g._ppl_no_ve = 50.0
    g._ppl_with_ve = 49.8
    g._ab_gain = 0.01
    g._ratio_ci = (0.98, 1.01)
    g._scales = {}
    g._target_modules = {}
    g._calibration_stats = {"status": "complete"}
    g._predictive_gate_state = {"passed": True, "evaluated": True}
    g._ab_seed_used = 999  # seed mismatch
    g._ab_windows_used = 4

    # Gate rejects; keep enabled_after_ab False to avoid errors
    monkeypatch.setattr(g, "_evaluate_ab_gate", lambda: (False, "below_threshold"))
    monkeypatch.setattr(g, "enable", lambda m: False)
    monkeypatch.setattr(g, "disable", lambda m: None)
    monkeypatch.setattr(g, "_refresh_after_edit_metrics", lambda m: None)

    res = g.finalize(nn.Linear(2, 2))
    warnings = ", ".join(res.get("warnings", []))
    assert "unexpected seed" in warnings
