import torch
import torch.nn as nn

from invarlock.guards.variance import equalise_residual_variance


class Proj(nn.Module):
    def __init__(self):
        super().__init__()
        # Provide weight param so scaling can be applied
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = None

    def forward(self, x):  # returns a Tensor; hook will see this
        # Produce scaled values so mean_square > 1
        return x.float() * 1.5


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_proj = Proj()

    def forward(self, x):
        return self.c_proj(x)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_proj = Proj()

    def forward(self, x):
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.mlp = MLP()

    def forward(self, x):
        a = self.attn(x)
        m = self.mlp(x)
        return a + m


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([Block()])
        # Add a parameter to supply a device for equalise_residual_variance
        self.dummy = nn.Linear(1, 1, bias=False)

    def forward(self, input_ids):
        # Expect a tensor; create a float tensor pathway
        x = input_ids.float()
        out = self.transformer.h[0](x)
        return out


def test_equalise_residual_variance_winsorize_and_chunks():
    model = TinyModel().eval()
    # 10 batches to trigger winsorization path
    dl = [{"input_ids": torch.ones(1, 4, dtype=torch.long)} for _ in range(10)]
    scales = equalise_residual_variance(
        model, dl, windows=10, tol=0.01, allow_empty=False
    )
    # Should produce scales for at least one branch (attn/mlp)
    assert isinstance(scales, dict) and len(scales) >= 1
