import pytest
import torch
import torch.nn as nn

import invarlock.guards.rmt_detection as rmt_detection


def test_apply_rmt_correction_scales_tied_params(monkeypatch):
    # Ensure Conv1D symbol exists for isinstance checks
    monkeypatch.setattr(rmt_detection, "Conv1D", nn.Linear, raising=False)

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

    rmt_detection._apply_rmt_correction(
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


def test_apply_rmt_correction_conv1d_and_multi_param_loop_paths(caplog):
    transformers_pytorch_utils = pytest.importorskip("transformers.pytorch_utils")
    Conv1D = transformers_pytorch_utils.Conv1D

    layer = Conv1D(4, 4)
    with torch.no_grad():
        layer.weight.mul_(50.0)

    tied_param = torch.nn.Parameter(torch.ones_like(layer.weight))

    class Adapter:
        def get_tying_map(self):
            return {"L0.weight": ["alias.weight"]}

        def get_parameter_by_name(self, name):
            if name == "alias.weight":
                return tied_param
            return None

    with caplog.at_level("INFO", logger=rmt_detection.__name__):
        rmt_detection._apply_rmt_correction(
            layer,
            factor=0.9,
            baseline_sigmas={"L0": 1.0},
            baseline_mp_stats={"L0": {"sigma_base": 1.0, "mp_bulk_edge_base": 1.0}},
            layer_name="L0",
            verbose=True,
            adapter=Adapter(),
        )
    assert "tied to 1 params" in caplog.text
    assert torch.allclose(tied_param, torch.ones_like(tied_param)) is False

    class _TwoWeightLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))
            self.aux_weight = nn.Parameter(torch.eye(2))

    skip_layer = _TwoWeightLayer()
    rmt_detection._apply_rmt_correction(
        skip_layer,
        factor=0.9,
        layer_name="skip",
        verbose=True,
    )

    original_svdvals = torch.linalg.svdvals
    calls = {"count": 0}

    def _svdvals(*args, **kwargs):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("boom")
        return original_svdvals(*args, **kwargs)

    failing_layer = _TwoWeightLayer()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.linalg, "svdvals", _svdvals)
        rmt_detection._apply_rmt_correction(
            failing_layer,
            factor=0.9,
            layer_name="fallback",
            verbose=False,
        )
