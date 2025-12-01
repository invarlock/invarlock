import torch

import invarlock.guards.spectral as spectral


def test_normalize_family_caps_numeric_and_default_false():
    # Numeric shorthand becomes mapping with kappa
    caps = spectral._normalize_family_caps({"ffn": 3.3})
    assert caps["ffn"]["kappa"] == 3.3
    # default=False yields empty mapping for invalid input
    assert spectral._normalize_family_caps(None, default=False) == {}


def test_compute_sigma_max_quantized_int8_skips():
    W = torch.zeros(2, 2, dtype=torch.int8)
    assert spectral.compute_sigma_max(W) == 1.0


def test_should_process_module_scope_ffn_proj():
    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.zeros(2, 2)

    m = Mod()
    assert spectral._should_process_module("layer.c_proj", m, "ffn+proj") is True
    assert spectral._should_process_module("layer.attn.c_proj", m, "attn") is True


def test_spectral_prepare_with_aliases(monkeypatch):
    # Patch heavy functions to avoid real tensor ops
    monkeypatch.setattr(
        "invarlock.guards.spectral.capture_baseline_sigmas", lambda *a, **k: {"m": 1.0}
    )
    monkeypatch.setattr(
        "invarlock.guards.spectral.classify_model_families",
        lambda *a, **k: {"m": "ffn"},
    )
    monkeypatch.setattr(
        "invarlock.guards.spectral.compute_family_stats",
        lambda *a, **k: {"ffn": {"mean": 1.0, "std": 0.0}},
    )
    monkeypatch.setattr(
        "invarlock.guards.spectral.scan_model_gains", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        "invarlock.guards.spectral.auto_sigma_target", lambda *a, **k: 1.0
    )

    g = spectral.SpectralGuard()
    policy = {
        # alias keys should be normalized
        "contraction": 0.9,
        "multipletesting": {"method": "bh", "alpha": 0.05, "m": 4},
        "baseline_family_stats": {"ffn": {"mean": 1.0, "std": 0.0}},
    }
    out = g.prepare(object(), object(), None, policy)
    assert out["ready"] is True
    assert g.config["sigma_quantile"] == 0.9
    assert "multiple_testing" in g.config
