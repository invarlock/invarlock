from types import SimpleNamespace

import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SimpleNamespace(c_proj=nn.Linear(4, 4, bias=False))
        self.mlp = SimpleNamespace(c_proj=nn.Linear(4, 4, bias=False))


class TinyTransformer(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.transformer = SimpleNamespace(h=[TinyBlock() for _ in range(layers)])


def _prepare(policy):
    g = VarianceGuard(policy=policy)
    g._prepared = True
    return g


def test_target_resolution_ffn_scope_default_tap():
    model = TinyTransformer(layers=2)
    g = _prepare({"scope": "ffn", "min_gain": 0.0})
    # prepare() populates targets via _resolve_target_modules
    g.prepare(model, adapter=None, calib=None)
    names = g._stats.get("target_module_names", [])
    assert all(".mlp.c_proj" in n for n in names)
    assert len(names) == 2


def test_target_resolution_attn_scope_with_tap():
    model = TinyTransformer(layers=2)
    g = _prepare(
        {"scope": "attn", "tap": ["transformer.h.*.attn.c_proj"], "min_gain": 0.0}
    )
    g.prepare(model, adapter=None, calib=None)
    names = g._stats.get("target_module_names", [])
    assert all(".attn.c_proj" in n for n in names)
    assert len(names) == 2


def test_target_resolution_both_scope_focus_modules():
    model = TinyTransformer(layers=2)
    focus = ["transformer.h.1.mlp.c_proj"]
    g = _prepare(
        {
            "scope": "both",
            "tap": ["transformer.h.*.mlp.c_proj", "transformer.h.*.attn.c_proj"],
            "target_modules": focus,
            "min_gain": 0.0,
        }
    )
    g.prepare(model, adapter=None, calib=None)
    names = g._stats.get("target_module_names", [])
    assert names == focus
