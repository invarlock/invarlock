import torch
import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


class IllCondLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Start well-conditioned; degeneracy is introduced post-prepare.
        self.weight = nn.Parameter(torch.randn(8, 8))


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
        correction_enabled=False,
        ignore_preview_inflation=False,
    )
    out = guard.prepare(model, adapter=None, calib=None, policy={})
    assert out["ready"] is True
    with torch.no_grad():
        model.transformer.h[0].attn.weight[-1].zero_()
    guard.after_edit(model)
    assert any(v["type"] == "degeneracy_norm_collapse" for v in guard.violations)
