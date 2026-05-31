from types import SimpleNamespace

import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class WeightObj:
    def dim(self):
        raise RuntimeError("no dim")

    @property
    def ndim(self):
        return 2


class WeirdModule:
    def __init__(self):
        self.weight = WeightObj()


class TinyWeird(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        blk = SimpleNamespace()
        blk.attn = SimpleNamespace(c_proj=WeirdModule())
        blk.mlp = SimpleNamespace(c_proj=nn.Linear(2, 2, bias=False))
        self.transformer.h = [blk]


def test_supported_module_when_weight_dim_raises_uses_ndim():
    g = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.attn.c_proj", "transformer.h.*.mlp.c_proj"],
        }
    )
    targets = g._resolve_target_modules(TinyWeird(), adapter=None)
    # attn.c_proj picked thanks to ndim fallback
    assert any("attn.c_proj" in k for k in targets.keys())
