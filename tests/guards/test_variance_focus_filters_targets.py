from types import SimpleNamespace

import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class TinyBoth(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        blk = SimpleNamespace()
        blk.attn = SimpleNamespace(c_proj=nn.Linear(2, 2, bias=False))
        blk.mlp = SimpleNamespace(c_proj=nn.Linear(2, 2, bias=False))
        self.transformer.h = [blk]


def test_focus_filters_targets_to_subset():
    g = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.attn.c_proj", "transformer.h.*.mlp.c_proj"],
        }
    )
    # Focus on attn only
    g._focus_modules = {g._normalize_module_name("transformer.h.0.attn.c_proj")}
    targets = g._resolve_target_modules(TinyBoth(), adapter=None)
    assert isinstance(targets, dict) and len(targets) == 1
    assert list(targets.keys())[0].endswith("attn.c_proj")
