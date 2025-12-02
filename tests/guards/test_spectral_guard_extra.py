import math
from typing import Any

import pytest
import torch

from invarlock.guards.spectral import (
    SpectralGuard,
    _normalize_family_caps,
    _summarize_sigmas,
    apply_relative_spectral_cap,
    apply_spectral_control,
    compute_z_score_for_value,
)


class DummyModule:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight


class DummyModel:
    def __init__(self, modules: dict[str, DummyModule]):
        self._modules = modules

    def named_modules(self):
        yield from self._modules.items()


def test_spectral_guard_before_edit_skips_when_not_prepared():
    guard = SpectralGuard()
    guard.prepared = False
    model = DummyModel({})

    guard.before_edit(model)

    assert any(event["operation"] == "before_edit_skipped" for event in guard.events)


def test_spectral_guard_detects_multiple_violation_types():
    guard = SpectralGuard(scope="all", correction_enabled=False, max_spectral_norm=1.0)
    guard.min_condition_number = 1e-2
    guard.deadband = 0.1
    guard.ignore_preview_inflation = False
    guard.ignore_preview_inflation = False
    guard.prepared = True
    guard.baseline_sigmas = {"layer.mlp.c_fc": 1.0}
    guard.baseline_family_stats = {"ffn": {"mean": 1.0, "std": 0.0}}
    guard.module_family_map = {"layer.mlp.c_fc": "ffn"}
    guard.family_caps = {"ffn": {"kappa": 0.5}}
    guard.target_sigma = 1.0

    weight = torch.tensor([[3.0, 0.0], [0.0, 1e-5]], dtype=torch.float32)
    module = DummyModule(weight)
    model = DummyModel({"layer.mlp.c_fc": module})

    metrics = {"layer.mlp.c_fc": 3.0}

    violations = guard._detect_spectral_violations(model, metrics, phase="after_edit")

    violation_types = {v["type"] for v in violations}
    assert "family_z_cap" in violation_types
    assert "max_spectral_norm" in violation_types
    assert "ill_conditioned" in violation_types
    assert guard.latest_z_scores["layer.mlp.c_fc"] > 0


def test_spectral_guard_validate_auto_prepare(monkeypatch):
    guard = SpectralGuard(scope="ffn", correction_enabled=True)
    guard.min_condition_number = 0.0

    def fake_capture(model: Any, scope: str):
        return {"layer.mlp.c_fc": 1.2}

    def fake_classify(model: Any, scope: str, existing=None):
        return {"layer.mlp.c_fc": "ffn"}

    def fake_family_stats(sigmas, family_map):
        return {"ffn": {"mean": 1.0, "std": 0.1}}

    monkeypatch.setattr(
        "invarlock.guards.spectral.capture_baseline_sigmas", fake_capture
    )
    monkeypatch.setattr(
        "invarlock.guards.spectral.classify_model_families", fake_classify
    )
    monkeypatch.setattr(
        "invarlock.guards.spectral.compute_family_stats", fake_family_stats
    )
    monkeypatch.setattr("invarlock.guards.spectral.scan_model_gains", lambda *_: {})
    monkeypatch.setattr("invarlock.guards.spectral.auto_sigma_target", lambda *_: 1.0)
    monkeypatch.setattr(
        "invarlock.guards.spectral.apply_relative_spectral_cap",
        lambda *_, **__: {"applied": False, "capped_modules": []},
    )
    monkeypatch.setattr(
        "invarlock.guards.spectral.apply_weight_rescale",
        lambda *_, **__: {"applied": False, "rescaled_modules": []},
    )
    monkeypatch.setattr(
        "invarlock.guards.spectral.apply_spectral_control",
        lambda *_, **__: {"applied": True, "capped_modules": ["layer.mlp.c_fc"]},
    )

    def fake_detect(self, model, metrics, phase="validate"):
        return [{"type": "mock_violation", "module": "layer.mlp.c_fc"}]

    monkeypatch.setattr(
        SpectralGuard, "_detect_spectral_violations", fake_detect, raising=False
    )

    module = DummyModule(torch.eye(2))
    model = DummyModel({"layer.mlp.c_fc": module})

    result = guard.validate(model, adapter=None, context={})

    assert result["action"] in {"warn", "abort"}
    assert isinstance(result["violations"], list)


def test_apply_relative_spectral_cap_scales_module():
    module = DummyModule(torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
    model = DummyModel({"layer": module})

    result = apply_relative_spectral_cap(
        model,
        cap_ratio=1.2,
        scope="all",
        baseline_sigmas={"layer": 1.0},
    )

    assert result["applied"] is True
    capped = result["capped_modules"][0]
    assert math.isclose(capped["scale_factor"], 0.6, rel_tol=1e-5)
    assert torch.allclose(module.weight, torch.tensor([[1.2, 0.0], [0.0, 0.6]]))


def test_apply_relative_spectral_cap_handles_failure(monkeypatch):
    module = DummyModule(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
    model = DummyModel({"layer": module})

    def raise_error(*_, **__):
        raise RuntimeError("boom")

    monkeypatch.setattr("invarlock.guards.spectral.compute_sigma_max", raise_error)

    result = apply_relative_spectral_cap(
        model,
        cap_ratio=1.0,
        scope="all",
        baseline_sigmas={"layer": 1.0},
    )

    assert result["applied"] is False
    assert result["failed_modules"]


def test_apply_spectral_control_rescale_branch(monkeypatch):
    model = DummyModel({})

    monkeypatch.setattr(
        "invarlock.guards.spectral.apply_relative_spectral_cap",
        lambda *_, **__: {"applied": False, "capped_modules": [], "failed_modules": []},
    )
    monkeypatch.setattr(
        "invarlock.guards.spectral.apply_weight_rescale",
        lambda *_, **__: {"applied": True, "rescaled_modules": ["layer"]},
    )

    policy = {"rescale_factor": 0.5, "scope": "all"}
    result = apply_spectral_control(model, policy)

    assert result["rescaling_applied"] is True
    assert result["applied"] is True


def test_normalize_family_caps_handles_invalid_entries():
    caps = {"ffn": {"kappa": 2.3}, "attn": 2.6, "embed": {"kappa": float("nan")}}
    normalized = _normalize_family_caps(caps, default=True)
    assert normalized["ffn"]["kappa"] == pytest.approx(2.3)
    assert normalized["attn"]["kappa"] == pytest.approx(2.6)
    assert "embed" not in normalized  # NaN dropped

    minimal = _normalize_family_caps({}, default=False)
    assert minimal == {}


def test_normalize_family_caps_returns_default_for_invalid():
    default_caps = _normalize_family_caps("invalid", default=True)
    assert default_caps["ffn"]["kappa"] == pytest.approx(2.5)


def test_compute_z_score_for_value_with_std():
    z = compute_z_score_for_value(
        1.5, {"mean": 1.0, "std": 0.25}, fallback_value=1.0, deadband=0.1
    )
    assert z == pytest.approx(2.0)


def test_compute_z_score_for_value_deadband_zero_std():
    z_zero = compute_z_score_for_value(
        1.04, {"mean": 1.0, "std": 0.0}, fallback_value=1.0, deadband=0.1
    )
    assert z_zero == 0.0

    z_scaled = compute_z_score_for_value(
        1.3, {"mean": 1.0, "std": 0.0}, fallback_value=1.0, deadband=0.1
    )
    assert z_scaled == pytest.approx(3.0)


def test_summarize_sigmas_handles_empty_and_values():
    empty_summary = _summarize_sigmas({})
    assert empty_summary["max_spectral_norm"] == 0.0

    summary = _summarize_sigmas({"a": 1.0, "b": 3.0})
    assert summary["max_spectral_norm"] == pytest.approx(3.0)
    assert summary["mean_spectral_norm"] == pytest.approx(2.0)
