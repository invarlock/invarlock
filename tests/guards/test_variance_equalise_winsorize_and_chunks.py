import torch
import torch.nn as nn

from invarlock.guards.variance import equalise_residual_variance


class TinyBlock(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(d, d, bias=True)
        self.mlp = nn.Module()
        self.mlp.c_proj = nn.Linear(d, d, bias=True)


class TinyModel(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock(d)])

    def forward(self, x):
        # Exercise both hooks; produce varied outputs to engage winsorization
        y1 = self.transformer.h[0].attn.c_proj(x)
        y2 = self.transformer.h[0].mlp.c_proj(x)
        return y1 + y2


def test_equalise_residual_variance_winsorize_and_chunks_applies_scales():
    model = TinyModel()
    # 12 small random batches to trigger quantile + chunk median path
    batches = [torch.randn(2, 4) for _ in range(12)]
    # Very small tolerance ensures we apply some scaling; keep clamp_range default
    scales = equalise_residual_variance(
        model, batches, windows=12, tol=0.0, clamp_range=(0.9, 1.1)
    )
    assert isinstance(scales, dict)
    # At least one branch should have a recorded scale
    assert any(k.endswith("attn") or k.endswith("mlp") for k in scales.keys())
