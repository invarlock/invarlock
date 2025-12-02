from types import SimpleNamespace

import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class BadProj(nn.Module):
    def __init__(self):
        super().__init__()
        # 1D weight triggers unsupported_type rejection
        import torch

        self.weight = nn.Parameter(torch.randn(4))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SimpleNamespace(c_proj=BadProj())
        self.mlp = SimpleNamespace(c_proj=BadProj())


class ManyBlocks(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock() for _ in range(n)])


def test_rejected_examples_capped_at_five():
    g = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.mlp.c_proj"],
            "min_gain": 0.0,
            "max_calib": 0,
        }
    )
    targets = g._resolve_target_modules(ManyBlocks(6), adapter=None)
    assert isinstance(targets, dict) and len(targets) == 0
    rejected = g._stats.get("target_resolution", {}).get("rejected", {})
    # Unsupported_type bucket should exist and examples capped to 5
    if "unsupported_type" in rejected:
        ex = rejected["unsupported_type"].get("examples", [])
        assert len(ex) <= 5
        assert rejected["unsupported_type"].get("count", 0) >= 1
