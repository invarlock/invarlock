import torch
import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


class IllCondLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Rank-deficient weight to yield very small min singular value
        W = torch.randn(8, 8)
        W[-1] = 0  # make last row zero to reduce rank
        self.weight = nn.Parameter(W)


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Module()])
        # Provide modules with weight to be checked by spectral guard
        self.transformer.h[0].attn = IllCondLayer()
        self.transformer.h[0].mlp = IllCondLayer()

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        yield from super().named_modules(memo, prefix, remove_duplicate)
        yield ("transformer.h.0.attn.c_proj", self.transformer.h[0].attn)
        yield ("transformer.h.0.mlp.c_fc", self.transformer.h[0].mlp)


def test_ill_conditioned_violation_detected():
    model = MiniModel()
    guard = SpectralGuard(
        min_condition_number=1e3,
        correction_enabled=False,
        ignore_preview_inflation=False,
    )
    out = guard.prepare(model, adapter=None, calib=None, policy={})
    assert out["ready"] is True
    guard.after_edit(model)
    assert any(v["type"] == "ill_conditioned" for v in guard.violations)
