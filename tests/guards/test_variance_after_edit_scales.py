import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        blk = nn.Module()
        blk.attn = nn.Module()
        blk.attn.c_proj = nn.Linear(2, 2, bias=False)
        blk.mlp = nn.Module()
        blk.mlp.c_proj = nn.Linear(2, 2, bias=False)
        self.transformer.h = nn.ModuleList([blk])


def test_after_edit_records_post_edit_scales():
    g = VarianceGuard()
    m = Tiny()
    g._prepared = True
    g._target_modules = g._resolve_target_modules(m, adapter=None)
    # Provide a scale so normalized_post_scales is non-empty
    any_name = next(iter(g._target_modules.keys()))
    g._scales = {any_name: 0.97}
    g.after_edit(m)
    # After-edit event emitted
    assert any(e.get("operation") == "after_edit" for e in g.events)
