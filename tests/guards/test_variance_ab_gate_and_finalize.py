import torch.nn as nn

from invarlock.guards.variance import VarianceGuard, _predictive_gate_outcome


class TinyBlock(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_proj = nn.Linear(d, d, bias=False)


class TinyModel(nn.Module):
    def __init__(self, n=1, d=4):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock(d) for _ in range(n)])

    def forward(self, x):
        # Simple path to exercise forward hooks if needed
        for blk in self.transformer.h:
            x = blk.attn.c_proj(x)
            x = blk.mlp.c_proj(x)
        return x


def _prime_guard_with_targets(g: VarianceGuard, model: nn.Module):
    targets = g._resolve_target_modules(model, adapter=None)
    assert targets, "no targets resolved"
    g._prepared = True
    g._target_modules = targets
    name = next(iter(targets.keys()))
    return name, targets[name]


def test_finalize_deadband_and_absolute_floor_errors(monkeypatch):
    model = TinyModel()
    g = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0, "max_calib": 0})
    _prime_guard_with_targets(g, model)

    # Force VE already enabled
    g._enabled = True
    # Synthetic A/B values that violate deadband and absolute floor
    g._ab_gain = 0.001  # below tie-breaker deadband (default 0.005)
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 99.99  # improvement = 0.01 < 0.05 absolute floor
    g._ratio_ci = (0.9, 0.95)

    # Monkeypatch AB gate to claim should_enable despite insufficient gain
    monkeypatch.setattr(g, "_evaluate_ab_gate", lambda: (True, "forced"))

    out = g.validate(model, adapter=None, context={})
    assert out["passed"] is False
    errs = "\n".join(out.get("errors", []))
    assert "tie-breaker deadband" in errs
    assert "absolute floor" in errs


def test_ab_gate_missing_and_invalid_ratio_ci_paths():
    model = TinyModel()
    g = VarianceGuard(
        policy={
            "scope": "ffn",
            "min_gain": 0.0,
            "max_calib": 0,
            "predictive_gate": False,
        }
    )
    _prime_guard_with_targets(g, model)

    # Missing ratio_ci
    g._ab_gain = 0.1
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 90.0
    g._ratio_ci = None
    res1 = g.validate(model, adapter=None, context={})
    assert isinstance(res1, dict)

    # Invalid ratio_ci
    g._ratio_ci = (0.0, -1.0)
    res2 = g.validate(model, adapter=None, context={})
    assert isinstance(res2, dict)


def test_monitor_only_sets_action_warn():
    model = TinyModel()
    g = VarianceGuard(
        policy={
            "scope": "ffn",
            "min_gain": 0.0,
            "max_calib": 0,
            "monitor_only": True,
            "predictive_gate": False,
        }
    )
    _prime_guard_with_targets(g, model)
    g._ab_gain = 0.2
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 80.0
    g._ratio_ci = (0.7, 0.9)
    out = g.validate(model, adapter=None, context={})
    assert out.get("action") == "warn"


def test_partial_enable_disable_paths():
    model = TinyModel()
    g = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0, "max_calib": 0})
    name, module = _prime_guard_with_targets(g, model)
    # Valid and invalid scale names
    g._scales = {name: 0.9, "transformer.h.9.mlp.c_proj": 1.1}
    assert g.enable(model) is True
    assert g.disable(model) is True


def test_enable_failure_checkpoint_pop_failed():
    model = TinyModel()
    g = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0, "max_calib": 0})
    g._prepared = True
    g._target_modules = {}  # no modules to checkpoint
    g._scales = {"transformer.h.99.mlp.c_proj": 1.1}  # nothing will apply
    assert g.enable(model) is False


def test_set_ab_results_manual_override_predictive_gate():
    g = VarianceGuard(policy={"scope": "ffn", "min_gain": 0.0, "max_calib": 0})
    g.set_ab_results(
        ppl_no_ve=100.0,
        ppl_with_ve=90.0,
        windows_used=1,
        seed_used=123,
        ratio_ci=(0.8, 0.9),
    )
    state = g._predictive_gate_state
    assert state.get("evaluated") is True and state.get("passed") is True
    assert state.get("reason") == "manual_override"


def test_predictive_gate_outcome_branches():
    # ci_unavailable
    ok, reason = _predictive_gate_outcome(0.0, None, min_effect=0.0, one_sided=True)
    assert ok is False and reason == "ci_unavailable"
    # one-sided: lower >= 0
    ok, reason = _predictive_gate_outcome(
        -0.1, (0.0, 0.1), min_effect=0.0, one_sided=True
    )
    assert ok is False and reason == "ci_contains_zero"
    # one-sided: mean_not_negative
    ok, reason = _predictive_gate_outcome(
        0.1, (-0.2, -0.01), min_effect=0.0, one_sided=True
    )
    assert ok is False and reason == "mean_not_negative"
    # one-sided: gain_below_threshold
    ok, reason = _predictive_gate_outcome(
        -0.02, (-0.2, -0.01), min_effect=0.05, one_sided=True
    )
    assert ok is False and reason == "gain_below_threshold"
    # one-sided: pass
    ok, reason = _predictive_gate_outcome(
        -0.1, (-0.2, -0.05), min_effect=0.01, one_sided=True
    )
    assert ok is True
    # two-sided: upper >= 0
    ok, reason = _predictive_gate_outcome(
        -0.1, (-0.2, 0.01), min_effect=0.0, one_sided=False
    )
    assert ok is False and reason == "ci_contains_zero"
    # two-sided: pass
    ok, reason = _predictive_gate_outcome(
        -0.1, (-0.2, -0.01), min_effect=0.0, one_sided=False
    )
    assert ok is True
