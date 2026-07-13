import numpy as np
import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_ab_gate_below_min_effect_lognll():
    g = VarianceGuard(policy={"min_effect_lognll": 0.05, "min_gain": 0.0})
    # Valid PPLs and ratio_ci
    g.set_ab_results(ppl_no_ve=100.0, ppl_with_ve=99.0, ratio_ci=(0.8, 0.9))
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("below_min_effect_lognll")


def test_finalize_monitor_only_and_post_edit_skipped(monkeypatch):
    g = VarianceGuard()
    g._prepared = True
    g._monitor_only = True
    # Force not evaluated yet and no calibration batches → skip path
    g._post_edit_evaluated = False
    g._calibration_batches = []

    # Ensure targets to avoid prepare failure branches
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

    out = g.finalize(Tiny())
    assert isinstance(out, dict)
    # monitor_only disables VE
    assert out["metrics"]["ve_enabled"] is False


def test_tensorize_and_ensure_tensor_value_numpy_and_metadata_ids():
    g = VarianceGuard()
    # np.ndarray should be converted to tensor
    arr = np.array([[1, 2, 3]], dtype=np.int64)
    t = g._ensure_tensor_value(arr)
    assert isinstance(t, torch.Tensor)
    # Extract window_ids from metadata field
    batches = [
        {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "metadata": {"window_ids": ["a", "b"]},
        }
    ]
    ids = g._extract_window_ids(batches)
    assert ids == ["a", "b"]


def test_ab_gate_invalid_ratio_ci_and_set_ab_predictive_update():
    g = VarianceGuard()
    # Invalid ratio_ci values (non-positive)
    g.set_ab_results(ppl_no_ve=100.0, ppl_with_ve=90.0, ratio_ci=(0.0, -1.0))
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason == "invalid_ratio_ci"

    # Now set a valid ratio_ci with upper < 1.0 to trigger predictive_gate_state update in set_ab_results
    g2 = VarianceGuard()
    g2.set_ab_results(ppl_no_ve=100.0, ppl_with_ve=90.0, ratio_ci=(0.8, 0.9))
    state = g2._predictive_gate_state
    assert state.get("evaluated") is True and state.get("passed") is True
