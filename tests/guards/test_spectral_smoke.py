import torch
import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard, capture_baseline_sigmas


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


def test_capture_baseline_sigmas_smoke() -> None:
    model = TinyModel([TinyLayer(), TinyLayer()])
    sigmas = capture_baseline_sigmas(model, scope="all")
    assert isinstance(sigmas, dict) and sigmas
    assert all(isinstance(v, float) and v >= 0.0 for v in sigmas.values())


def test_spectral_finalize_passes_without_changes() -> None:
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    try:
        model = TinyModel([TinyLayer(), TinyLayer()])
        guard = SpectralGuard()
        out = guard.prepare(model, adapter=None, calib=None, policy={})
        assert out["ready"] is True

        result = guard.finalize(model)
        assert result["passed"] is True
        assert result["metrics"]["caps_applied"] == 0
        assert result["metrics"]["caps_exceeded"] is False
    finally:
        torch.random.set_rng_state(rng_state)


def test_spectral_finalize_fails_when_budget_exceeded() -> None:
    model = TinyModel([TinyLayer(), TinyLayer()])
    guard = SpectralGuard(max_caps=0)
    out = guard.prepare(model, adapter=None, calib=None, policy={})
    assert out["ready"] is True

    # Ensure any detected budgeted violation is selected by the multiple-testing filter.
    guard.multiple_testing = {"method": "bonferroni", "alpha": 1.0, "m": 1}

    # Induce a budgeted violation by scaling a single module beyond deadband.
    with torch.no_grad():
        model.layers[0].attn_c_proj.weight.mul_(2.0)

    # Make the cap intentionally strict so a nonzero z-score triggers a budgeted violation.
    guard.family_caps["attn"]["kappa"] = 0.0

    result = guard.finalize(model)
    assert result["passed"] is False
    assert result["metrics"]["caps_applied"] >= 1
    assert result["metrics"]["caps_exceeded"] is True


def test_spectral_finalize_allows_equal_caps_budget() -> None:
    model = TinyModel([TinyLayer()])
    # Our TinyLayer yields family-alias module names (e.g., "attn.c_proj"), so a single
    # underlying module may appear more than once in named_modules(). Use max_caps=2
    # and assert the equality boundary (caps_applied == max_caps) passes.
    guard = SpectralGuard(max_caps=2, degeneracy={"enabled": False})
    out = guard.prepare(model, adapter=None, calib=None, policy={})
    assert out["ready"] is True

    guard.multiple_testing = {"method": "bonferroni", "alpha": 1.0, "m": 1}
    guard.family_caps["attn"]["kappa"] = 0.0
    guard.family_caps["ffn"]["kappa"] = 1e6

    with torch.no_grad():
        model.layers[0].attn_c_proj.weight.mul_(2.0)

    result = guard.finalize(model)
    assert result["passed"] is True
    assert result["metrics"]["caps_applied"] == 2
    assert result["metrics"]["caps_exceeded"] is False


def test_spectral_finalize_fails_on_max_spectral_norm() -> None:
    model = TinyModel([TinyLayer()])
    guard = SpectralGuard(max_spectral_norm=0.01, degeneracy={"enabled": False})
    out = guard.prepare(model, adapter=None, calib=None, policy={})
    assert out["ready"] is True

    result = guard.finalize(model)
    assert result["passed"] is False
    assert any(v.get("type") == "max_spectral_norm" for v in result["violations"])
    assert result["metrics"]["fatal_violations"] >= 1


def test_spectral_finalize_fails_on_fatal_violation_severity(monkeypatch) -> None:
    model = TinyModel([TinyLayer()])
    guard = SpectralGuard(degeneracy={"enabled": False})
    out = guard.prepare(model, adapter=None, calib=None, policy={})
    assert out["ready"] is True

    def _fatal(_model, _metrics, phase: str = "finalize"):
        _ = phase
        return [
            {
                "type": "degeneracy_norm_collapse",
                "severity": "fatal",
                "module": "attn.c_proj",
                "family": "attn",
                "message": "fatal violation",
            }
        ]

    monkeypatch.setattr(guard, "_detect_spectral_violations", _fatal, raising=True)

    result = guard.finalize(model)
    assert result["passed"] is False
    assert result["metrics"]["fatal_violations"] >= 1
    assert "fatal violation" in result["errors"]
