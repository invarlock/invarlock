import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_expected_window_ids_and_monitor_only_warning(monkeypatch):
    g = VarianceGuard()
    g._pairing_reference = ["preview::1", "preview::2"]
    assert g._expected_window_ids() == ["preview::1", "preview::2"]

    # Monitor-only finalize path should emit coverage warning when status != complete
    g = VarianceGuard(policy={"min_gain": 0.0, "scope": "both", "max_calib": 0})
    g._prepared = True
    g._monitor_only = True
    g._enabled = False
    g._target_modules = {"transformer.h.0.mlp.c_proj": nn.Linear(2, 2, bias=False)}
    g._ppl_no_ve = 50.0
    g._ppl_with_ve = 49.0
    g._ab_gain = 0.02
    g._ratio_ci = (0.9, 0.95)
    g._calibration_stats = {"status": "incomplete"}
    monkeypatch.setattr(g, "_evaluate_ab_gate", lambda: (False, "below_threshold"))
    monkeypatch.setattr(g, "_refresh_after_edit_metrics", lambda m: None)
    res = g.finalize(nn.Linear(2, 2))
    assert any("monitor mode" in w for w in res.get("warnings", []))
