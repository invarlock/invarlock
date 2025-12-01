import torch
import torch.nn as nn

from invarlock.guards.variance import equalise_residual_variance


class TinyBlock(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_proj = nn.Linear(d, d, bias=False)


class TinyModel(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock(d)])

    def forward(self, x):
        # Exercise both branches
        y1 = self.transformer.h[0].attn.c_proj(x)
        y2 = self.transformer.h[0].mlp.c_proj(x)
        return y1 + y2


def test_equalise_residual_variance_quantiles_and_chunks():
    model = TinyModel()
    # 16 batches triggers quantile winsorization and chunked median path
    batches = [torch.randn(2, 4) for _ in range(16)]
    scales = equalise_residual_variance(
        model, batches, windows=16, tol=0.0, clamp_range=(0.8, 1.2)
    )
    # At least one branch should have a computed scale
    assert isinstance(scales, dict) and any(scales.values())
