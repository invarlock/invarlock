import pytest
import torch
import torch.nn as nn

from invarlock.guards.spectral import apply_weight_rescale, auto_sigma_target


def test_apply_weight_rescale_scales_linear_modules() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    original_weight = model[0].weight.clone()

    result = apply_weight_rescale(model, scale_factor=0.5, scope="all")

    assert result["applied"] is True
    assert result["rescaled_modules"], "Expected at least one module to be rescaled"
    assert torch.allclose(model[0].weight, original_weight * 0.5)


def test_auto_sigma_target_fallback() -> None:
    class EmptyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    target = auto_sigma_target(EmptyModel(), kappa=0.9)
    assert target == pytest.approx(0.9, rel=1e-6)
