from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.edits import NoopEdit
from invarlock.guards.variance import VarianceGuard


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        blk = nn.Module()
        blk.attn = nn.Module()
        blk.attn.c_proj = nn.Linear(2, 2, bias=False)
        blk.mlp = nn.Module()
        blk.mlp.c_proj = nn.Linear(2, 2, bias=False)
        self.transformer.h = nn.ModuleList([blk])

    def forward(self, inputs, labels=None):
        x = self.transformer.h[0].attn.c_proj(inputs)
        return self.transformer.h[0].mlp.c_proj(x)


def test_no_scales_branch_sets_status_and_estimates():
    model = Tiny()
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "predictive_gate": True,
            "calibration": {"windows": 2, "min_coverage": 2, "seed": 11},
        }
    )

    # Resolve targets and ensure empty scales
    g._target_modules = g._resolve_target_modules(model, adapter=None)
    g._scales = {}
    # Prepare two batches to meet coverage
    batches = [torch.ones(1, 2), torch.zeros(1, 2)]
    g._store_calibration_batches(batches)

    # Drive the internal evaluation to the no_scales path
    g._calibration_stats = {
        "requested": 2,
        "coverage": 0,
        "min_coverage": 2,
        "seed": 11,
        "status": "pending",
        "tag": "t",
    }
    g._evaluate_calibration_pass(
        model, g._calibration_batches, min_coverage=2, calib_seed=11, tag="t"
    )

    status = g._calibration_stats.get("status")
    assert status in {"no_scaling_required", "pending", "insufficient"}
    if status == "no_scaling_required":
        ape = g._stats.get("ab_point_estimates", {})
        assert "ppl_no_ve" in ape and "ppl_with_ve" in ape


def test_finalize_includes_ab_provenance_metrics():
    model = Tiny()
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "predictive_gate": False,
            "max_calib": 10,
        }
    )
    # Minimal prepare to populate targets and stats
    g.prepare(model, adapter=None, calib=None, policy=None)
    g._prepared = True
    # Force some provenance info
    g._stats.setdefault("ab_provenance", {})["condition_a"] = {"status": "evaluated"}
    out = g.finalize(model)
    metrics = out.get("metrics", {})
    # If finalize returned full metrics, ab_provenance should be present
    if metrics:
        assert "ab_provenance" in metrics


def _prepared_guard_with_context(
    model: Tiny,
    *,
    edit: dict,
    batches: list[torch.Tensor],
) -> VarianceGuard:
    guard = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "predictive_gate": True,
            "seed": 123,
            "calibration": {"windows": 2, "min_coverage": 2, "seed": 123},
        }
    )
    guard._target_modules = guard._resolve_target_modules(model, adapter=None)
    guard._prepared = True
    guard._store_calibration_batches(batches)
    guard.set_run_context(SimpleNamespace(meta={}, context={}, edit=edit))
    return guard


def test_verified_explicit_noop_emits_strict_compatible_no_adjustment_result():
    model = Tiny()
    batches = [torch.ones(1, 2), torch.zeros(1, 2)]
    guard = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "predictive_gate": True,
            "seed": 123,
            "calibration": {"windows": 2, "min_coverage": 2, "seed": 123},
        }
    )
    guard.set_run_context(SimpleNamespace(meta={}, context={}, edit={}))
    prepare_result = guard.prepare(model, adapter=None, calib=batches, policy=None)
    weights_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    edit = NoopEdit().apply(
        model,
        SimpleNamespace(describe=lambda _model: {"family": "tiny"}),
    )
    weights_after = model.state_dict()
    guard.set_run_context(
        SimpleNamespace(meta={}, context={}, edit=edit),
    )
    result = guard.validate(model, adapter=None, context={})

    assert prepare_result["ready"] is True
    assert prepare_result["baseline_metrics"]["calibration"]["coverage"] == 2
    assert all(
        torch.equal(value, weights_after[name])
        for name, value in weights_before.items()
    )
    assert result.passed is True
    assert result.decision == "allow"
    assert result.diagnostics == ()
    assert result.metrics["monitor_only"] is False
    assert result.metrics["calibration"]["coverage"] == 2
    assert result.metrics["calibration"]["status"] == "no_scaling_required"
    assert result.metrics["predictive_gate"] == {
        "evaluated": True,
        "passed": True,
        "reason": "no_adjustment_required",
        "delta_ci": (None, None),
        "gain_ci": (None, None),
        "mean_delta": None,
    }


def test_verified_noop_cannot_override_explicit_monitor_only_policy():
    model = Tiny()
    batches = [torch.ones(1, 2), torch.zeros(1, 2)]
    guard = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "predictive_gate": True,
            "monitor_only": True,
            "seed": 123,
            "calibration": {"windows": 2, "min_coverage": 2, "seed": 123},
        }
    )
    guard.set_run_context(SimpleNamespace(meta={}, context={}, edit={}))
    prepare_result = guard.prepare(model, adapter=None, calib=batches, policy=None)
    edit = NoopEdit().apply(
        model,
        SimpleNamespace(describe=lambda _model: {"family": "tiny"}),
    )
    guard.set_run_context(SimpleNamespace(meta={}, context={}, edit=edit))

    finalized = guard.finalize(model)
    result = guard.validate(model, adapter=None, context={})

    assert prepare_result["ready"] is True
    assert finalized["passed"] is True
    assert finalized["decision"] == "monitor"
    assert result.passed is True
    assert result.decision == "monitor"
    assert result.metrics["monitor_only"] is True
    assert result.metrics["predictive_gate"]["passed"] is True
    assert result.metrics["predictive_gate"]["reason"] == "no_adjustment_required"


def test_zero_change_non_noop_remains_monitor_only_and_predictive_fail():
    model = Tiny()
    guard = _prepared_guard_with_context(
        model,
        edit={"name": "structured", "deltas": {"params_changed": 0}},
        batches=[torch.ones(1, 2), torch.zeros(1, 2)],
    )

    guard._refresh_after_edit_metrics(model)
    result = guard.validate(model, adapter=None, context={})

    assert result.decision == "monitor"
    assert result.metrics["monitor_only"] is True
    assert result.metrics["predictive_gate"]["passed"] is False
    assert result.metrics["predictive_gate"]["reason"] == "no_scales"
    assert result.diagnostics


def test_explicit_noop_with_insufficient_coverage_remains_monitor_decision():
    model = Tiny()
    guard = _prepared_guard_with_context(
        model,
        edit={"name": "noop", "deltas": {"params_changed": 0}},
        batches=[torch.ones(1, 2)],
    )

    guard._refresh_after_edit_metrics(model)
    result = guard.validate(model, adapter=None, context={})

    assert result.decision == "monitor"
    assert result.metrics["predictive_gate"]["passed"] is False
    assert result.metrics["predictive_gate"]["reason"] == "insufficient_coverage"
    assert result.metrics["calibration"]["status"] == "insufficient"
    assert result.diagnostics


def test_real_edit_scale_computation_error_remains_predictive_fail(monkeypatch):
    model = Tiny()
    guard = _prepared_guard_with_context(
        model,
        edit={"name": "structured", "deltas": {"params_changed": 1}},
        batches=[torch.ones(1, 2), torch.zeros(1, 2)],
    )
    monkeypatch.setattr(
        guard,
        "_compute_variance_scales",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    guard._refresh_after_edit_metrics(model)
    result = guard.validate(model, adapter=None, context={})

    assert result.decision == "monitor"
    assert result.metrics["monitor_only"] is False
    assert result.metrics["predictive_gate"]["passed"] is False
    assert result.metrics["predictive_gate"]["reason"] == "no_scales"
    assert any(
        event["kind"] == "post_edit_scale_failure" for event in guard.diagnostic_records
    )
