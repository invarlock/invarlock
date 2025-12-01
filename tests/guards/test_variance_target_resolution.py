import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class BadModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 1D weight to fail the dim==2 heuristic
        self.weight = nn.Parameter(torch.randn(4))


class TinyBlock(nn.Module):
    def __init__(self, good: bool = True, with_attn: bool = True):
        super().__init__()
        self.attn = nn.Module()
        self.mlp = nn.Module()
        if with_attn:
            self.attn.c_proj = nn.Linear(4, 4, bias=False)
        if good:
            self.mlp.c_proj = nn.Linear(4, 4, bias=False)
        else:
            self.mlp.c_proj = BadModule()


class TinyModel(nn.Module):
    def __init__(self, good: bool = True, with_attn: bool = True):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock(good=good, with_attn=with_attn)])


def test_rejections_cover_unsupported_type_for_mlp():
    # Force mlp to be considered and rejected for unsupported type
    m = TinyModel(good=False, with_attn=True)
    g = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.mlp.c_proj"],
            "min_gain": 0.0,
            "max_calib": 0,
        }
    )
    res = g.prepare(m, adapter=None, calib=None, policy=None)
    assert res["ready"] in (True, False)
    tr = g._stats.get("target_resolution", {})
    rejected = tr.get("rejected", {})
    assert "unsupported_type" in rejected


def test_rejections_cover_tap_mismatch_for_mlp():
    # Configure tap so only attn matches; mlp becomes tap_mismatch
    m = TinyModel(good=False, with_attn=True)
    g = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.attn.c_proj"],
            "min_gain": 0.0,
            "max_calib": 0,
        }
    )
    res = g.prepare(m, adapter=None, calib=None, policy=None)
    assert res["ready"] in (True, False)
    tr = g._stats.get("target_resolution", {})
    rejected = tr.get("rejected", {})
    assert "tap_mismatch" in rejected


def test_adapter_error_bucket_present_when_adapter_fails():
    class BadAdapter:
        def get_layer_modules(self, model, idx):
            raise RuntimeError("boom")

    # Force no direct targets so adapter fallback is used
    m = TinyModel(good=True, with_attn=False)
    g = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.attn.c_proj"],
            "min_gain": 0.0,
            "max_calib": 0,
        }
    )
    g.prepare(m, adapter=BadAdapter(), calib=None, policy=None)
    tr = g._stats.get("target_resolution", {})
    rejected = tr.get("rejected", {})
    assert any(reason.startswith("adapter_error:") for reason in rejected.keys())
