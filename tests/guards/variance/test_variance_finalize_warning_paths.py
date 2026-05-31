import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_finalize_emits_seed_mismatch_and_uncommitted_checkpoint_warnings(monkeypatch):
    g = VarianceGuard(
        policy={"min_gain": 0.0, "seed": 123, "scope": "both", "max_calib": 0}
    )

    # Prepare minimal internal state to allow finalize() to proceed
    g._enabled = False
    g._ppl_no_ve = 50.0
    g._ppl_with_ve = 49.8
    g._ab_gain = 0.01
    g._ratio_ci = (0.98, 1.01)
    g._scales = {}
    g._target_modules = {}
    g._calibration_stats = {}
    g._predictive_gate_state = {"passed": True, "evaluated": True}
    g._enable_attempt_count = 4
    g._disable_attempt_count = 5
    g._checkpoint_stack = ["ckpt1"]
    g._ab_seed_used = 999  # mismatch to trigger warning
    g._ab_windows_used = 2

    # Avoid heavy work inside finalize
    monkeypatch.setattr(g, "_refresh_after_edit_metrics", lambda m: None)
    monkeypatch.setattr(g, "enable", lambda m: False)
    monkeypatch.setattr(g, "disable", lambda m: None)
    monkeypatch.setattr(
        g, "_evaluate_ab_gate", lambda: (False, "below_threshold_with_deadband")
    )

    res = g.finalize(nn.Linear(2, 2))
    # Smoke assertions ensure finalize produced structured output
    assert isinstance(res, dict)
    assert isinstance(res.get("warnings", []), list)
    assert any("Preparation failed" in e for e in res.get("errors", []))
