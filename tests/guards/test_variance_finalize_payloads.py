import torch.nn as nn

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

    def forward(self, x):
        x = self.transformer.h[0].attn.c_proj(x)
        x = self.transformer.h[0].mlp.c_proj(x)
        return x


def test_finalize_includes_expected_metrics_payloads():
    model = TinyModel()
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 0,
            "predictive_gate": False,
        }
    )
    # Prepare without calibration to populate pre-edit stats
    prep = g.prepare(model, adapter=None, calib=None, policy=None)
    assert isinstance(prep, dict) and "baseline_metrics" in prep

    # Let finalize run _refresh_after_edit_metrics to populate post-edit fields
    out = g.finalize(model)
    assert isinstance(out, dict)
    metrics = out.get("metrics", {})
    # Check for key payloads regardless of emptiness
    for key in (
        "target_module_names",
        "tap",
        "ab_provenance",
        "proposed_scales_pre_edit",
        "proposed_scales_post_edit",
        "raw_scales_pre_edit",
        "raw_scales_post_edit",
    ):
        assert key in metrics


def test_finalize_warns_on_uncommitted_checkpoints():
    model = TinyModel()
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 0,
            "predictive_gate": False,
        }
    )
    # Resolve targets and mark prepared
    targets = g._resolve_target_modules(model, adapter=None)
    g._prepared = True
    g._target_modules = targets
    # Push a checkpoint without committing to trigger warning
    g._push_checkpoint(model)
    # Provide minimal A/B fields
    g._ab_gain = 0.0
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 100.0
    g._ratio_ci = None
    # Finalize should include warning about uncommitted checkpoints
    res = g.finalize(model)
    warnings = "\n".join(res.get("warnings", []))
    assert "Uncommitted checkpoints remaining" in warnings


def test_finalize_errors_on_deadband_and_abs_floor(monkeypatch):
    model = TinyModel()
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 0,
            "predictive_gate": False,
        }
    )
    # Mark prepared and minimal targets
    g._prepared = True
    g._target_modules = g._resolve_target_modules(model, adapter=None)
    # Set A/B results such that VE is enabled but thresholds are not met
    g._enabled = True
    g._ab_gain = 0.0005  # Below tie-breaker deadband (0.001)
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 99.97  # Improvement 0.03 < abs floor 0.05
    g._ratio_ci = (0.9, 0.95)
    # Force gate approval
    monkeypatch.setattr(g, "_evaluate_ab_gate", lambda: (True, "forced"))
    out = g.finalize(model)
    errors = "\n".join(out.get("errors", []))
    assert "tie-breaker deadband" in errors
    assert "absolute floor" in errors
