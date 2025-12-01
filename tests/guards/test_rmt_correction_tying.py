import torch
import torch.nn as nn

import invarlock.guards.rmt as R


def test_apply_rmt_correction_scales_tied_params(monkeypatch):
    # Ensure Conv1D symbol exists for isinstance checks
    monkeypatch.setattr(R, "Conv1D", nn.Linear, raising=False)

    layer = nn.Linear(8, 8)
    with torch.no_grad():
        layer.weight.mul_(3.0)

    # Track a tied parameter value
    tied_param = torch.nn.Parameter(torch.ones_like(layer.weight))

    class Adapter:
        def get_tying_map(self):
            return {"L0.weight": ["alias.weight"]}

        def get_parameter_by_name(self, name):
            if name == "alias.weight":
                return tied_param
            return None

    baseline_sigmas = {"L0": 1.0}
    baseline_mp_stats = {"L0": {"sigma_base": 1.0, "mp_bulk_edge_base": 1.0}}

    pre_layer_sigma = torch.linalg.svdvals(layer.weight.float()).max().item()
    pre_tied = tied_param.detach().clone()

    R._apply_rmt_correction(
        layer,
        factor=0.9,
        baseline_sigmas=baseline_sigmas,
        baseline_mp_stats=baseline_mp_stats,
        layer_name="L0",
        deadband=0.0,
        verbose=False,
        adapter=Adapter(),
    )

    post_layer_sigma = torch.linalg.svdvals(layer.weight.float()).max().item()
    # Ensure layer sigma decreases and tied parameter changed
    assert post_layer_sigma <= pre_layer_sigma
    assert not torch.allclose(tied_param, pre_tied)
