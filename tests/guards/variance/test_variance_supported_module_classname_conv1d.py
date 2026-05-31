from types import SimpleNamespace

import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class Conv1D:
    # Not a real nn.Module, but has class name Conv1D
    def __init__(self):
        pass


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        blk = SimpleNamespace()
        blk.attn = SimpleNamespace(c_proj=Conv1D())
        blk.mlp = SimpleNamespace(c_proj=nn.Linear(2, 2, bias=False))
        self.transformer.h = [blk]

    def named_modules(self):
        yield ("root", nn.ReLU())


def test_supported_module_detected_by_classname_conv1d():
    g = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.attn.c_proj", "transformer.h.*.mlp.c_proj"],
        }
    )
    targets = g._resolve_target_modules(TinyModel(), adapter=None)
    # Both attn (FakeConv1D by classname) and mlp (Linear) should be picked
    assert any("attn.c_proj" in k for k in targets.keys())
    assert any("mlp.c_proj" in k for k in targets.keys())
