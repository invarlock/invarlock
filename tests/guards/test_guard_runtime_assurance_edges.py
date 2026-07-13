from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from invarlock.guards import rmt_runtime, spectral_runtime, variance_ops


def _rmt_guard(*, required: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        name="rmt",
        estimator={"type": "activation_edge"},
        activation_sampling={"windows": {"count": 2}},
        _external_baseline_required=required,
        _external_baseline_ready=False,
        _external_baseline_reason=None,
        _external_baseline_evidence=None,
        _require_activation=True,
        _activation_ready=True,
        _activation_required_failed=False,
        _activation_required_reason=None,
        prepared=True,
        baseline_edge_risk_by_family={"ffn": 1.0},
        baseline_edge_risk_by_module={"layer": 1.0},
        edge_risk_by_family={"ffn": 1.0},
        edge_risk_by_module={"layer": 1.0},
        epsilon_by_family={"ffn": 0.1},
        epsilon_violations=[],
        _calibration_batches=[],
    )


def _rmt_evidence(guard: SimpleNamespace) -> dict[str, Any]:
    return {
        "metrics": {
            "measurement_contract": rmt_runtime._rmt_measurement_contract(guard),
            "edge_risk_by_family": {"ffn": 2.0},
            "edge_risk_by_module": {"layer": 2.0},
        }
    }


def test_rmt_external_baseline_evidence_fails_closed_for_each_missing_anchor() -> None:
    guard = _rmt_guard(required=False)
    assert rmt_runtime.load_external_baseline_evidence(guard) == {
        "ready": False,
        "required": False,
        "reason": "not_required",
    }

    guard = _rmt_guard()
    assert rmt_runtime.load_external_baseline_evidence(guard)["reason"] == (
        "baseline_rmt_evidence_missing"
    )

    guard._external_baseline_evidence = {"metrics": {}}
    assert rmt_runtime.load_external_baseline_evidence(guard)["reason"] == (
        "baseline_rmt_measurement_contract_mismatch"
    )

    guard._external_baseline_evidence = {
        "metrics": {
            "measurement_contract": rmt_runtime._rmt_measurement_contract(guard)
        }
    }
    assert rmt_runtime.load_external_baseline_evidence(guard)["reason"] == (
        "baseline_rmt_family_measurements_missing"
    )

    guard._external_baseline_evidence["metrics"]["edge_risk_by_family"] = {"ffn": 1.0}
    assert rmt_runtime.load_external_baseline_evidence(guard)["reason"] == (
        "baseline_rmt_module_measurements_missing"
    )


def test_rmt_external_baseline_rejects_nonfinite_and_coverage_mismatch() -> None:
    guard = _rmt_guard()
    evidence = _rmt_evidence(guard)
    evidence["metrics"]["edge_risk_by_family"] = {
        "ffn": "invalid",
        "negative": -1,
        "infinite": float("inf"),
    }
    guard._external_baseline_evidence = evidence
    result = rmt_runtime.load_external_baseline_evidence(guard)
    assert result["reason"] == "baseline_rmt_family_coverage_mismatch"
    assert not guard._external_baseline_ready

    guard = _rmt_guard()
    evidence = _rmt_evidence(guard)
    evidence["metrics"]["edge_risk_by_module"] = {"other": 2.0}
    guard._external_baseline_evidence = evidence
    result = rmt_runtime.load_external_baseline_evidence(guard)
    assert result["reason"] == "baseline_rmt_module_coverage_mismatch"


def test_rmt_external_baseline_success_replaces_local_reference_exactly() -> None:
    guard = _rmt_guard()
    guard._external_baseline_evidence = _rmt_evidence(guard)

    result = rmt_runtime.load_external_baseline_evidence(guard)

    assert result["ready"] is True
    assert result["families"] == result["modules"] == 1
    assert guard.baseline_edge_risk_by_family == {"ffn": 2.0}
    assert guard.baseline_edge_risk_by_module == {"layer": 2.0}
    assert guard._external_baseline_ready is True


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            {"_external_baseline_ready": False, "_external_baseline_reason": None},
            "baseline_rmt_evidence_unavailable",
        ),
        (
            {"edge_risk_by_family": {}, "edge_risk_by_module": {}},
            "no_activation_edge_measurements",
        ),
        (
            {"edge_risk_by_family": {"other": 1.0}},
            "subject_rmt_measurement_coverage_mismatch",
        ),
    ],
)
def test_rmt_finalize_blocks_unanchored_or_incomparable_measurements(
    mutation: dict[str, Any], reason: str
) -> None:
    guard = _rmt_guard()
    guard._external_baseline_ready = True
    for key, value in mutation.items():
        setattr(guard, key, value)

    result = rmt_runtime.finalize_rmt_guard(
        guard,
        model=object(),
        has_guard_outcome=False,
        guard_outcome_type=None,
    )

    assert result["passed"] is False
    assert result["decision"] == "block"
    assert result["metrics"]["unsupported_reason"] == reason
    assert result["violations"][0]["severity"] == "error"


def test_rmt_unsupported_outcome_preserves_typed_guard_result_shape() -> None:
    guard = _rmt_guard()
    result = rmt_runtime.finalize_rmt_guard(
        guard,
        model=object(),
        has_guard_outcome=True,
        guard_outcome_type=lambda **values: SimpleNamespace(**values),
    )

    assert result.passed is False
    assert result.decision == "block"
    assert result.violations[0]["type"] == "rmt_unsupported"
    assert result.metrics["unsupported_reason"] == ("baseline_rmt_evidence_unavailable")


def _spectral_guard(*, required: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        estimator={"type": "power_iter", "iters": 2},
        degeneracy={"enabled": False},
        _external_baseline_required=required,
        _external_baseline_ready=False,
        _external_baseline_reason=None,
        _external_baseline_evidence=None,
        baseline_sigmas={"layer": 1.0},
        module_family_map={"layer": "ffn"},
        baseline_family_stats={},
        baseline_degeneracy={},
        baseline_metrics={},
        _serialize_policy=lambda: {"mode": "strict"},
    )


def _spectral_evidence(guard: SimpleNamespace) -> dict[str, Any]:
    return {
        "metrics": {},
        "baseline_metrics": {
            "module_sigmas": {"layer": 2.0},
            "measurement_contract": spectral_runtime._spectral_measurement_contract(
                guard
            ),
            "baseline_degeneracy": {"layer": {"stable_rank": 2.0}, "bad": []},
        },
        "final_metrics": {},
        "module_family_map": {"layer": "ffn"},
    }


def test_spectral_external_baseline_fails_closed_matrix() -> None:
    guard = _spectral_guard(required=False)
    assert spectral_runtime.load_external_baseline_evidence(guard)["reason"] == (
        "not_required"
    )

    guard = _spectral_guard()
    assert spectral_runtime.load_external_baseline_evidence(guard)["reason"] == (
        "baseline_spectral_evidence_missing"
    )
    guard._external_baseline_evidence = {}
    assert spectral_runtime.load_external_baseline_evidence(guard)["reason"] == (
        "baseline_spectral_module_measurements_missing"
    )

    evidence = _spectral_evidence(guard)
    evidence["baseline_metrics"]["module_sigmas"] = {
        "layer": "invalid",
        "negative": -1,
        "infinite": float("inf"),
    }
    guard._external_baseline_evidence = evidence
    result = spectral_runtime.load_external_baseline_evidence(guard)
    assert result["reason"] == "baseline_spectral_module_coverage_mismatch"
    assert result["missing"] == ["layer"]

    guard = _spectral_guard()
    evidence = _spectral_evidence(guard)
    evidence["baseline_metrics"]["measurement_contract"] = {"wrong": True}
    guard._external_baseline_evidence = evidence
    assert spectral_runtime.load_external_baseline_evidence(guard)["reason"] == (
        "baseline_spectral_measurement_contract_mismatch"
    )

    guard = _spectral_guard()
    evidence = _spectral_evidence(guard)
    evidence["module_family_map"] = []
    guard._external_baseline_evidence = evidence
    assert spectral_runtime.load_external_baseline_evidence(guard)["reason"] == (
        "baseline_spectral_family_map_missing"
    )

    guard = _spectral_guard()
    evidence = _spectral_evidence(guard)
    evidence["module_family_map"] = {"layer": ""}
    guard._external_baseline_evidence = evidence
    assert spectral_runtime.load_external_baseline_evidence(guard)["reason"] == (
        "baseline_spectral_family_coverage_mismatch"
    )


def test_spectral_external_baseline_success_replaces_reference() -> None:
    guard = _spectral_guard()
    guard._external_baseline_evidence = _spectral_evidence(guard)

    result = spectral_runtime.load_external_baseline_evidence(guard)

    assert result["ready"] is True
    assert guard.baseline_sigmas == {"layer": 2.0}
    assert guard.module_family_map == {"layer": "ffn"}
    assert guard.baseline_degeneracy == {"layer": {"stable_rank": 2.0}}
    assert guard.baseline_metrics["module_sigmas"] == {"layer": 2.0}
    assert guard._external_baseline_ready is True


def _spectral_validation_guard() -> SimpleNamespace:
    guard = _spectral_guard()
    guard.prepared = True
    guard.deadband = 0.0
    guard.max_caps = 1
    guard.sigma_quantile = 0.9
    guard.family_caps = {}
    guard.multiple_testing = {"method": "bh", "alpha": 0.05}
    guard.latest_z_scores = {}
    guard._measurement_diagnostics = []
    guard._capture_sigmas = lambda model, phase: {"layer": 2.0}
    guard._detect_spectral_violations = lambda model, metrics, phase: []
    guard._select_budgeted_violations = lambda values: (list(values), {})
    return guard


def test_spectral_validation_blocks_missing_external_and_subject_coverage() -> None:
    guard = _spectral_validation_guard()
    guard._capture_sigmas = lambda model, phase: {}
    result = spectral_runtime.validate_guard(guard, object(), None, {})
    assert result.extras["reason"] == "no_eligible_modules_measured"

    guard = _spectral_validation_guard()
    guard._external_baseline_ready = False
    result = spectral_runtime.validate_guard(guard, object(), None, {})
    assert result.extras["reason"] == "baseline_spectral_evidence_unavailable"

    guard = _spectral_validation_guard()
    guard._external_baseline_ready = True
    guard._capture_sigmas = lambda model, phase: {"other": 2.0}
    result = spectral_runtime.validate_guard(guard, object(), None, {})
    assert result.extras["reason"] == "subject_spectral_module_coverage_mismatch"
    assert guard._external_baseline_ready is False


def test_spectral_strict_external_selected_violation_is_always_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _spectral_validation_guard()
    guard._external_baseline_ready = True
    violation = {
        "type": "sigma_drift",
        "severity": "warning",
        "message": "selected paired regression",
        "module": "layer",
    }
    guard._detect_spectral_violations = lambda model, metrics, phase: [violation]
    monkeypatch.setattr(
        spectral_runtime,
        "partition_spectral_violations",
        lambda values: ([], list(values)),
    )
    monkeypatch.setattr(
        spectral_runtime,
        "evaluate_spectral_outcome",
        lambda **kwargs: {
            "selected_violations": [violation],
            "candidate_budgeted": 1,
            "caps_applied": 1,
            "caps_exceeded": False,
            "passed": True,
            "decision": "allow",
        },
    )

    result = spectral_runtime.validate_guard(guard, object(), None, {})

    assert result.passed is False
    assert result.decision == "block"
    assert result.violations == (violation,)
    assert result.metrics["baseline_source"] == "external_run"


def _variance_guard() -> SimpleNamespace:
    events: list[tuple[str, dict[str, object]]] = []
    return SimpleNamespace(
        _target_modules={},
        _checkpoint_stack=[],
        _log_event=lambda operation, **data: events.append((operation, data)),
        _events=events,
        _enabled=True,
        _stats={},
        _disable_attempt_count=0,
        _enable_attempt_count=0,
        _scales={},
        _original_scales={},
        _prepared=True,
        _monitor_only=False,
        _scale_matches_target=lambda scale_name, target_name: scale_name == target_name,
    )


def test_variance_checkpoint_empty_and_shape_mismatch_do_not_mutate_state() -> None:
    guard = _variance_guard()
    guard._checkpoint_stack = [{}]
    assert variance_ops.pop_checkpoint(guard, model=None) is False
    assert guard._checkpoint_stack == [{}]

    module = torch.nn.Linear(3, 2, bias=False)
    guard._target_modules = {"layer": module}
    guard._checkpoint_stack = [{"layer": torch.ones(2, 2)}]
    before = module.weight.detach().clone()
    assert variance_ops.pop_checkpoint(guard, model=None) is False
    assert torch.equal(module.weight, before)
    assert guard._checkpoint_stack


def test_variance_checkpoint_copy_failure_preserves_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _variance_guard()
    module = torch.nn.Linear(2, 2, bias=False)
    guard._target_modules = {"layer": module}
    guard._checkpoint_stack = [{"layer": module.weight.detach().clone()}]
    monkeypatch.setattr(
        torch.Tensor,
        "copy_",
        lambda self, other: (_ for _ in ()).throw(RuntimeError("copy failed")),
    )
    assert variance_ops.pop_checkpoint(guard, model=None) is False
    assert guard._checkpoint_stack
    assert any(event == "checkpoint_pop_failed" for event, _ in guard._events)


def test_variance_enable_partial_failure_is_reported_without_false_full_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _variance_guard()
    guard._enabled = False
    good = torch.nn.Linear(2, 2, bias=False)
    missing = SimpleNamespace(weight=torch.ones(2, 2))
    guard._target_modules = {"good": good, "missing": missing}
    guard._scales = {"good": 2.0, "missing": 2.0}
    original_target = variance_ops._target_module_for_scale
    seen = 0

    def changing_target(current_guard: Any, scale_name: str) -> Any | None:
        nonlocal seen
        seen += 1
        # First four lookups are preflight/snapshot checks; make the second target
        # disappear only during application to exercise partial failure handling.
        if scale_name == "missing" and seen > 4:
            return None
        return original_target(current_guard, scale_name)

    monkeypatch.setattr(variance_ops, "_target_module_for_scale", changing_target)
    assert variance_ops.enable_guard(guard, model=None) is True
    assert guard._enabled is True
    assert any(event == "enable_partial" for event, _ in guard._events)


def test_variance_enable_catastrophic_completion_failure_rolls_back() -> None:
    guard = _variance_guard()
    guard._enabled = False
    module = torch.nn.Linear(2, 2, bias=False)
    original = module.weight.detach().clone()
    guard._target_modules = {"layer": module}
    guard._scales = {"layer": 2.0}
    events: list[tuple[str, dict[str, object]]] = []

    def log_event(operation: str, **data: object) -> None:
        if operation == "enable_complete":
            raise RuntimeError("completion failed")
        events.append((operation, data))

    guard._log_event = log_event
    guard._events = events
    assert variance_ops.enable_guard(guard, model=None) is False
    assert torch.equal(module.weight, original)
    assert any(event == "enable_catastrophic_failure" for event, _ in events)


def test_variance_disable_target_matching_and_partial_failure() -> None:
    guard = _variance_guard()
    exact = torch.nn.Linear(2, 2, bias=False)
    alias = torch.nn.Linear(2, 2, bias=False)
    guard._target_modules = {"exact": exact, "base": alias}
    guard._scales = {"exact": 2.0, "alias": 2.0, "missing": 2.0}
    guard._scale_matches_target = lambda scale, target: (
        scale == "alias" and target == "base"
    )
    with torch.no_grad():
        exact.weight.mul_(2)
        alias.weight.mul_(2)

    assert variance_ops.disable_guard(guard, model=None) is True
    assert any(event == "disable_partial" for event, _ in guard._events)
    assert any(event == "scale_reverted" for event, _ in guard._events)
