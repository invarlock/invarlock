import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        blk = nn.Module()
        blk.attn = nn.Module()
        blk.attn.c_proj = nn.Linear(2, 2, bias=False)
        blk.mlp = nn.Module()
        blk.mlp.c_proj = nn.Linear(2, 2, bias=False)
        self.transformer.h = nn.ModuleList([blk])

    def forward(self, x):
        return self.transformer.h[0].mlp.c_proj(self.transformer.h[0].attn.c_proj(x))


def test_before_edit_and_after_edit_branches():
    g = VarianceGuard(policy={"scope": "both", "min_gain": 0.0, "max_calib": 0})
    # before_edit logs only if prepared
    g._prepared = True
    g.before_edit(M())
    # after_edit path when not prepared
    g._prepared = False
    g.after_edit(M())
    # after_edit when prepared calls refresh metrics and logs
    g._prepared = True
    g.after_edit(M())
