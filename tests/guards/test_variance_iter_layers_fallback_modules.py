import torch.nn as nn

from invarlock.guards.variance import equalise_residual_variance


class HasProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_proj = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.c_proj(x)


class FallbackBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = HasProj()
        self.mlp = HasProj()

    def forward(self, x):
        return self.mlp(self.attn(x))


class FallbackModel(nn.Module):
    def __init__(self):
        super().__init__()
        # no transformer/model/encoder attributes to force fallback path
        self.block = FallbackBlock()

    def forward(self, x):
        return self.block(x)


def test_iter_transformer_layers_fallback_module_detection():
    model = FallbackModel()
    # Empty dataloader with allow_empty, small windows
    out = equalise_residual_variance(model, dataloader=[], windows=0, allow_empty=True)
    assert out == {}
