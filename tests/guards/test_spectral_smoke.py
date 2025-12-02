import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


class TinyLayer(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        # Names include family hints for classification
        self.attn_c_proj = nn.Linear(d, d)
        self.mlp_c_fc = nn.Linear(d, d)

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        # Provide names that mimic transformer families
        yield from super().named_modules(memo, prefix, remove_duplicate)
        yield ("attn.c_proj", self.attn_c_proj)
        yield ("mlp.c_fc", self.mlp_c_fc)


class TinyModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        yield from super().named_modules(memo, prefix, remove_duplicate)
        for i, layer in enumerate(self.layers):
            yield (f"transformer.h.{i}", layer)


def test_spectral_prepare_ready():
    model = TinyModel([TinyLayer(), TinyLayer()])
    guard = SpectralGuard()
    out = guard.prepare(model, adapter=None, calib=None, policy={"sigma_quantile": 0.9})
    assert out["ready"] is True
    assert "baseline_metrics" in out
    assert "target_sigma" in out
