import torch
import torch.nn as nn

import invarlock.guards.variance as var_mod
from invarlock.guards.variance import VarianceGuard


class TinyBlock(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_proj = nn.Linear(d, d, bias=False)


class TinyModel(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock(d)])

    def forward(self, inputs, labels=None):
        x = self.transformer.h[0].attn.c_proj(inputs)
        x = torch.tanh(x)
        x = self.transformer.h[0].mlp.c_proj(x)
        return x


def _batches(n=3, b=4, d=4):
    xs = []
    for _ in range(n):
        inputs = torch.randn(b, d)
        labels = torch.zeros(b, d)
        xs.append((inputs, labels))
    return xs


def test_evaluate_calibration_pass_with_enable_and_delta_ci_error(monkeypatch):
    g = VarianceGuard(
        policy={"scope": "both", "min_gain": 0.0, "max_calib": 20, "alpha": 0.05}
    )
    # Prepare target modules and a synthetic scale to allow enable path
    model = TinyModel()
    targets = g._resolve_target_modules(model, adapter=None)
    g._target_modules = targets
    # Fingerprint path exercised via target modules
    g._scales = {next(iter(targets.keys())): 0.95}

    # Enable/disable succeed without actually modifying state
    monkeypatch.setattr(g, "enable", lambda m: True)
    monkeypatch.setattr(g, "disable", lambda m: True)

    # Force compute_paired_delta_log_ci to raise to hit warn branch
    monkeypatch.setattr(
        var_mod,
        "compute_paired_delta_log_ci",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")),
    )

    batches = _batches(n=3)
    g._evaluate_calibration_pass(
        model, calibration_batches=batches, min_coverage=2, calib_seed=1337, tag="unit"
    )
    # Predictive gate state is recorded; after exception, delta_ci entry exists
    pg = g._stats.get("predictive_gate", {})
    assert "delta_ci" in pg
    # Provenance for both conditions recorded
    ab = g._stats.get("ab_provenance", {})
    assert "condition_a" in ab


def test_evaluate_calibration_pass_no_calibration_records_fingerprint():
    g = VarianceGuard(policy={"scope": "both"})
    model = TinyModel()
    # Populate target modules so fingerprint path runs
    g._target_modules = g._resolve_target_modules(model, adapter=None)
    g._evaluate_calibration_pass(
        model, calibration_batches=[], min_coverage=1, calib_seed=1, tag="empty"
    )
    # Stats should include predictive gate and possibly target_fingerprint
    assert "predictive_gate" in g._stats
    # target_fingerprint entry may exist when hashing succeeds
    assert "target_fingerprint" in g._stats
