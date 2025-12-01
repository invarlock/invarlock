import torch
import torch.nn as nn

import invarlock.guards.rmt as R


def test_apply_rmt_correction_scales_down(monkeypatch):
    # Ensure Conv1D symbol exists to avoid NameError in isinstance checks
    monkeypatch.setattr(R, "Conv1D", nn.Linear, raising=False)

    layer = nn.Linear(8, 8)
    with torch.no_grad():
        layer.weight.mul_(2.0)

    # Baseline stats with modest sigma_base so target < current
    baseline_sigmas = {"L0": 1.0}
    baseline_mp_stats = {"L0": {"sigma_base": 1.0, "mp_bulk_edge_base": 1.0}}

    # Pre-correction sigma
    pre = torch.linalg.svdvals(layer.weight.float()).max().item()
    R._apply_rmt_correction(
        layer,
        factor=0.9,
        baseline_sigmas=baseline_sigmas,
        baseline_mp_stats=baseline_mp_stats,
        layer_name="L0",
        deadband=0.0,
        verbose=False,
    )
    post = torch.linalg.svdvals(layer.weight.float()).max().item()
    assert post <= pre
