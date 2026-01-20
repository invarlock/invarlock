import torch.nn as nn

from invarlock.guards.variance import equalise_residual_variance


class DummyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_proj = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.c_proj(x)


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_proj = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.c_proj(x)


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyAttn()
        self.mlp = DummyMLP()


class ModelLayersStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyBlock()])


class BertStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([DummyBlock()])


def test_iter_layers_model_layers_and_bert_styles_allow_empty():
    # Both architectures should be iterable; allow_empty skips data collection
    out1 = equalise_residual_variance(
        ModelLayersStyle(), dataloader=[], allow_empty=True, windows=1
    )
    out2 = equalise_residual_variance(
        BertStyle(), dataloader=[], allow_empty=True, windows=1
    )
    assert out1 == {} and out2 == {}
