import torch
import torch.nn as nn

from invarlock.guards.variance import equalise_residual_variance


class AttnOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_proj = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.out_proj(x)


class MLPDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.down_proj(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = AttnOut()
        self.mlp = MLPDown()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([Block()])

    def forward(self, x):
        y = self.transformer.h[0].attn(x)
        y = self.transformer.h[0].mlp(y)
        return y


def test_equalise_uses_out_and_down_proj_with_clamp_none():
    model = Model()
    dataloader = [torch.ones(1, 2)]
    out = equalise_residual_variance(
        model, dataloader, windows=1, allow_empty=False, clamp_range=None
    )
    # Should run and return a mapping; keys come from out/down_proj paths
    assert isinstance(out, dict)
