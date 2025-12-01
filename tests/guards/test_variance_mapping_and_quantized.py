import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


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
        x = self.transformer.h[0].attn.c_proj(x)
        x = self.transformer.h[0].mlp.c_proj(x)
        return x


def test_enable_block_name_mapping_applies_scale():
    model = TinyModel()
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 0,
            "tap": ["transformer.h.*.attn.c_proj", "transformer.h.*.mlp.c_proj"],
        }
    )
    targets = g._resolve_target_modules(model, adapter=None)
    g._prepared = True
    g._target_modules = targets
    # Use block-style scale key to trigger mapping logic
    g._scales = {"block0.attn": 0.9}
    attn_name = next(k for k in targets if ".attn.c_proj" in k)
    attn_module = targets[attn_name]
    before = attn_module.weight.detach().clone()
    assert g.enable(model) is True
    after = attn_module.weight.detach().clone()
    assert not torch.allclose(before, after)
    assert g.disable(model) is True


class Int8LinearLike(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Create a weight attribute with int8 dtype and 2D shape
        self.weight = torch.zeros(out_features, in_features, dtype=torch.int8)

    def forward(self, x):
        return x


def test_quantized_weight_skip_in_enable():
    model = TinyModel()
    # Replace mlp c_proj with int8-like module to trigger skip
    model.transformer.h[0].mlp.c_proj = Int8LinearLike(4, 4)
    g = VarianceGuard(policy={"scope": "both", "min_gain": 0.0, "max_calib": 0})
    targets = g._resolve_target_modules(model, adapter=None)
    g._prepared = True
    g._target_modules = targets
    # Only target the mlp to ensure applied_count becomes 0 and enable fails
    quant_name = next(k for k in targets if ".mlp.c_proj" in k)
    g._scales = {quant_name: 1.1}
    assert g.enable(model) is False
    # Also exercise disable() revert skip path; mark enabled to bypass idempotent return
    g._enabled = True
    assert g.disable(model) is False
