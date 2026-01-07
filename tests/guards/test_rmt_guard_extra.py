import pytest
import torch
import torch.nn as nn

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
    _ = monkeypatch
    guard = RMTGuard()
    guard.prepared = True
    guard.baseline_edge_risk_by_family = {"attn": 1.0}
    guard.edge_risk_by_family = {"attn": 2.0}
    guard.epsilon_by_family = {"attn": 0.0}

    model = DummyModel(DummyLayer())
    result = guard.validate(model, adapter=None, context={})

    assert "edge_risk_by_family" in result["metrics"]
    assert result["metrics"]["epsilon_violations"]


def test_rmt_guard_set_epsilon_from_dict_and_scalar():
    guard = RMTGuard(
        epsilon_default=0.08, epsilon_by_family={"attn": 0.05, "ffn": "0.08", "invalid": "x"}  # type: ignore[arg-type]
    )
    assert guard.epsilon_by_family["attn"] == pytest.approx(0.05)
    assert guard.epsilon_by_family["ffn"] == pytest.approx(0.08)
    assert guard.epsilon_default == pytest.approx(0.08)
    # Scalar update should overwrite per-family defaults
    guard._set_epsilon_default(0.12)
    guard._set_epsilon_by_family({"attn": 0.12, "ffn": 0.12, "embed": 0.12, "other": 0.12})
    assert all(val == pytest.approx(0.12) for val in guard.epsilon_by_family.values())
    assert guard.epsilon_default == pytest.approx(0.12)


def test_rmt_guard_compute_epsilon_violations_detects_overages():
    guard = RMTGuard(epsilon_default=0.0)
    guard.baseline_edge_risk_by_family = {"attn": 1.0, "embed": 1.0}
    guard.edge_risk_by_family = {"attn": 2.0, "embed": 1.2}
    guard.epsilon_by_family["embed"] = 0.05

    violations = guard._compute_epsilon_violations()
    violation_families = {v["family"] for v in violations}
    assert violation_families == {"attn", "embed"}
    embed_violation = next(v for v in violations if v["family"] == "embed")
    assert embed_violation["allowed"] == pytest.approx(1.05)


def test_rmt_guard_compute_epsilon_violations_respects_allowed_threshold():
    guard = RMTGuard(epsilon_by_family={"attn": 0.5})
    guard.baseline_edge_risk_by_family = {"attn": 2.0}
    guard.edge_risk_by_family = {"attn": 3.0}

    violations = guard._compute_epsilon_violations()
    assert violations == []  # allowed = 2.0 * (1 + 0.5) = 3.0, so within limit
