import torch
import torch.nn as nn

from invarlock.guards.variance import equalise_residual_variance


class ZeroOutProj(nn.Linear):
    def __init__(self):
        super().__init__(2, 2, bias=False)

    def forward(self, x):  # type: ignore[override]
        # Produce zero-width output to trigger y.numel() == 0 in the forward hook
        y = super().forward(x)
        return y[:, :0]


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_proj = ZeroOutProj()

    def forward(self, x):
        # Match GPT-2-like structure expected by iter
        return self.c_proj(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # transformer.h[0].mlp.c_proj path used by the equaliser
        self.transformer = nn.Module()
        blk = nn.Module()
        blk.attn = TinyBlock()
        blk.mlp = TinyBlock()
        self.transformer.h = nn.ModuleList([blk])

    def forward(self, x):
        # Exercise both attn and mlp projections independently on same input
        _ = self.transformer.h[0].mlp(x)
        y = self.transformer.h[0].attn(x)
        return y


def test_equalise_residual_variance_hook_early_return_on_zero_output():
    model = TinyModel()
    dataloader = [torch.ones(1, 2)]
    # Should not crash; hooks see zero-sized outputs and skip; no scales applied
    out = equalise_residual_variance(model, dataloader, windows=1, allow_empty=False)
    assert out == {}
