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

    class DummyModel:
        def named_modules(self):
            return iter([])

    policy = {
        "sigma_quantile": 0.9,
        "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
        "estimator": {"iters": -1, "init": "e0"},
        "degeneracy": {
            "enabled": True,
            "stable_rank": {"warn_ratio": 0.75, "fatal_ratio": 0.5},
            "norm_collapse": {"warn_ratio": 0.5, "fatal_ratio": 0.25},
        },
        "baseline_family_stats": {"ffn": {"mean": 1.0, "std": 0.0}},
    }
    out = g.prepare(DummyModel(), object(), None, policy)
    assert out["ready"] is True
    assert g.config["sigma_quantile"] == 0.9
    assert "multiple_testing" in g.config
    assert g.estimator["iters"] == 1
    assert g.estimator["init"] == "e0"


def test_spectral_prepare_estimator_invalid_iters_and_init_defaults(
    monkeypatch,
) -> None:
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

    class DummyModel:
        def named_modules(self):
            return iter([])

    out = g.prepare(
        DummyModel(),
        object(),
        None,
        {"estimator": {"iters": "bad", "init": "bad"}},
    )
    assert out["ready"] is True
    assert g.estimator["iters"] == 4
    assert g.estimator["init"] == "ones"


def test_spectral_guard_init_parses_estimator_and_degeneracy_branches() -> None:
    g_default = spectral.SpectralGuard(
        estimator={"iters": "bad", "init": "bad"}, degeneracy="bad"
    )
    assert g_default.estimator["iters"] == 4
    assert g_default.estimator["init"] == "ones"

    g_min = spectral.SpectralGuard(estimator={"iters": -1, "init": "e0"})
    assert g_min.estimator["iters"] == 1
    assert g_min.estimator["init"] == "e0"

    g_dict = spectral.SpectralGuard(degeneracy={"enabled": False})
    assert g_dict.degeneracy["enabled"] is False


def test_spectral_set_run_context_captures_profile() -> None:
    g = spectral.SpectralGuard()
    report = type("R", (), {"context": {"profile": "CI"}})()
    g.set_run_context(report)
    assert g._run_profile == "ci"
    report2 = type("R2", (), {"context": ["not", "dict"]})()
    g.set_run_context(report2)
    assert g._run_profile is None


def test_compute_sigma_max_additional_branches(monkeypatch) -> None:
    assert spectral.compute_sigma_max("not_a_tensor", iters="bad", init="bad") == 1.0
    assert spectral.compute_sigma_max(torch.empty((0, 3)), iters=1) == 0.0
    assert spectral.compute_sigma_max(torch.zeros(3), iters=1) == 0.0

    sigma = spectral.compute_sigma_max(torch.eye(2), iters=0, init="bad")
    assert sigma > 0.0

    monkeypatch.setattr(
        "invarlock.guards.spectral.power_iter_sigma_max", lambda *_a, **_k: 1 / 0
    )
    assert spectral.compute_sigma_max(torch.eye(2), iters=1) == 1.0


def test_classify_module_family_moe_and_module_type_branches() -> None:
    linear = torch.nn.Linear(2, 2)
    assert spectral.classify_module_family("router.gate", linear) == "router"
    assert spectral.classify_module_family("experts.block", linear) == "expert_ffn"
    assert spectral.classify_module_family("layer.attn.c_proj", linear) == "attn"
    assert spectral.classify_module_family("layer.mlp.c_fc", linear) == "ffn"

    # module type based embedding classification
    assert (
        spectral.classify_module_family(
            "layer.not_named_embed", torch.nn.Embedding(2, 2)
        )
        == "embed"
    )

    class _WeightOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.zeros(2, 2)

    m = _WeightOnly()
    assert spectral._should_process_module("any.name", m, "unknown-scope") is True
