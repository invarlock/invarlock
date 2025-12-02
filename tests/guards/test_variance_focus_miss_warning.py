from types import SimpleNamespace

import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class OneBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        blk = SimpleNamespace()
        blk.attn = SimpleNamespace(c_proj=nn.Linear(2, 2, bias=False))
        blk.mlp = SimpleNamespace(c_proj=nn.Linear(2, 2, bias=False))
        self.transformer.h = [blk]

    def named_modules(self):
        # Minimal iterator is fine; _iter_transformer_layers doesn't rely on this here
        yield ("root", nn.ReLU())


def test_focus_miss_logs_and_filters():
    # Configure tap to allow matching, but focus to a module name that won't match
    g = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.attn.c_proj", "transformer.h.*.mlp.c_proj"],
        }
    )
    g._focus_modules = {"transformer.h.9.mlp.c_proj"}

    targets = g._resolve_target_modules(OneBlock(), adapter=None)
    # Even with focus miss, original matches are preserved (warning logged)
    assert isinstance(targets, dict) and len(targets) >= 1
    stats = g._stats.get("target_resolution", {})
    # target_resolution stats should reflect the matches
    assert stats.get("total_matched", 0) == len(targets)
