import torch.nn as nn

from invarlock.guards.spectral import (
    SpectralGuard,
    _normalize_family_caps,
    classify_module_family,
    scan_model_gains,
)


def test_normalize_family_caps_variants():
    # Empty or invalid returns defaults when default=True
    caps = _normalize_family_caps(None)
    assert isinstance(caps, dict) and caps
    # Dict with numeric and non-numeric
    caps2 = _normalize_family_caps({"ffn": {"kappa": 2.0, "x": "y"}}, default=True)
    assert caps2["ffn"]["kappa"] == 2.0


def test_classify_module_family():
    class Dummy:
        __name__ = "linear"

    assert classify_module_family("mlp.c_fc", Dummy()) == "ffn"
    assert classify_module_family("attn.c_proj", Dummy()) == "attn"
    assert classify_module_family("wte", Dummy()) == "embed"
    assert classify_module_family("other", Dummy()) in {"ffn", "attn", "embed", "other"}


def test_observability_and_scan_model_gains_smoke():
    class TinyLayer(nn.Module):
        def __init__(self, d=4):
            super().__init__()
            self.attn_c_proj = nn.Linear(d, d)
            self.mlp_c_fc = nn.Linear(d, d)

        def named_modules(self, memo=None, prefix="", remove_duplicate=True):
            yield from super().named_modules(memo, prefix, remove_duplicate)
            yield ("attn.c_proj", self.attn_c_proj)
            yield ("mlp.c_fc", self.mlp_c_fc)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])

        def named_modules(self, memo=None, prefix="", remove_duplicate=True):
            yield from super().named_modules(memo, prefix, remove_duplicate)
            for i, layer in enumerate(self.layers):
                yield (f"transformer.h.{i}", layer)

    model = TinyModel()
    guard = SpectralGuard()
    out = guard.prepare(model, adapter=None, calib=None, policy={})
    assert out["ready"] is True
    # seed latest z and exercise observability summaries
    guard.latest_z_scores = {
        "transformer.h.0.attn.c_proj": 1.2,
        "transformer.h.1.mlp.c_fc": 0.8,
    }
    fam, listings = guard._compute_family_observability()
    assert isinstance(fam, dict) and isinstance(listings, dict)

    gains = scan_model_gains(model)
    assert isinstance(gains, dict) and "scanned_modules" in gains
