import pytest
import torch
import torch.nn as nn

from invarlock.guards import rmt
from invarlock.guards.rmt import (
    RMTGuard,
    _apply_rmt_correction,
    rmt_detect_with_names,
)


class DummyAdapter:
    def __init__(self):
        self.scaled = None

    def get_tying_map(self):
        return {"layer.weight": ["layer.weight_tied"]}

    def get_parameter_by_name(self, name: str):
        if name == "layer.weight_tied":

            class TiedParam:
                def __init__(self, parent):
                    self.parent = parent

                def mul_(self, factor):
                    self.parent.scaled = factor

            return TiedParam(self)
        raise KeyError(name)


class DummyLayer(nn.Module):
    def __init__(self, scale: float = 10.0):
        super().__init__()
        self.attn = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.attn.weight.copy_(torch.eye(4) * scale)
            self.mlp.weight.copy_(torch.eye(4))


class DummyModel(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer0 = layer

    def named_modules(self):
        yield ("model", self)
        yield ("layer0", self.layer0)


def test_apply_rmt_correction_scales_weight_and_tied_param():
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 3.0]]))
    adapter = DummyAdapter()

    baseline_sigmas = {"layer": 1.0}
    baseline_mp = {"layer": {"sigma_base": 1.0}}

    _apply_rmt_correction(
        layer,
        factor=1.5,
        baseline_sigmas=baseline_sigmas,
        baseline_mp_stats=baseline_mp,
        layer_name="layer",
        deadband=0.1,
        verbose=False,
        adapter=adapter,
    )
    assert adapter.scaled is not None
    with torch.no_grad():
        sigma_after = torch.linalg.svdvals(layer.weight)[0].item()
    assert sigma_after < 3.0


def test_rmt_detect_with_names_fallback_path():
    model = DummyModel(DummyLayer())
    report = rmt_detect_with_names(model, threshold=0.9, verbose=False)

    assert report["has_outliers"] is True
    assert report["flagged_layers"]
    assert report["layers"]


def test_rmt_guard_validate_uses_detection(monkeypatch):
    guard = RMTGuard(deadband=0.0)
    guard.prepared = True
    guard.baseline_mp_stats = {"layer0": {"sigma_base": 1.0}}
    guard.baseline_sigmas = {"layer0.attn.weight": 1.0}
    guard.scope = "all"
    guard.config = {"deadband": 0.0, "margin": 1.5}
    guard.policy = guard.config
    guard.epsilon_by_family = {"attn": 0.1}
    guard.outliers_per_family = {"attn": 2}
    guard.baseline_outliers_per_family = {"attn": 1}
    guard.outliers_total = 2
    guard.baseline_total_outliers = 1

    def fake_detect(model, threshold=1.5, verbose=False):
        return {
            "has_outliers": True,
            "n_layers_flagged": 1,
            "outlier_count": 1,
            "max_ratio": 2.0,
            "threshold": 1.5,
            "flagged_layers": ["layer0"],
            "per_layer": [],
            "outliers": [],
            "epsilon_by_family": {"attn": 0.1},
            "families": {"attn": {"bare": 1, "guarded": 2}},
        }

    monkeypatch.setattr(rmt, "rmt_detect_with_names", fake_detect)

    model = DummyModel(DummyLayer())
    result = guard.validate(model, adapter=None, context={})

    assert "outliers_per_family" in result["metrics"]
    assert result["metrics"]["outliers_per_family"]["attn"] == 2


def test_rmt_guard_set_epsilon_from_dict_and_scalar():
    guard = RMTGuard(epsilon={"attn": 0.05, "ffn": "0.08", "invalid": "x"})
    assert guard.epsilon_by_family["attn"] == pytest.approx(0.05)
    assert guard.epsilon_by_family["ffn"] == pytest.approx(0.08)
    assert guard.epsilon_default == pytest.approx(0.08)
    # Scalar update should overwrite per-family defaults
    guard._set_epsilon(0.12)
    assert all(val == pytest.approx(0.12) for val in guard.epsilon_by_family.values())
    assert guard.epsilon_default == pytest.approx(0.12)


def test_rmt_guard_compute_epsilon_violations_detects_overages():
    guard = RMTGuard(epsilon=0.0)
    guard.baseline_outliers_per_family = {"attn": 1, "embed": 0}
    guard.outliers_per_family = {"attn": 2, "embed": 1}
    guard.epsilon_by_family["embed"] = 0.05

    violations = guard._compute_epsilon_violations()
    violation_families = {v["family"] for v in violations}
    assert violation_families == {"attn", "embed"}
    embed_violation = next(v for v in violations if v["family"] == "embed")
    assert embed_violation["allowed"] == 0  # Bare = 0 → allowed 0


def test_rmt_guard_compute_epsilon_violations_respects_allowed_threshold():
    guard = RMTGuard(epsilon={"attn": 0.5})
    guard.baseline_outliers_per_family = {"attn": 2}
    guard.outliers_per_family = {"attn": 3}

    violations = guard._compute_epsilon_violations()
    assert violations == []  # allowed = ceil(2 * 1.5) = 3, so within limit
