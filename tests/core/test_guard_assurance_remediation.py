from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn

from invarlock.core.api import EditRuntime, Guard, RunConfig, RunReport
from invarlock.core.runner_runtime.execution_plan import (
    RunnerExecutionRequest,
    RunnerExecutionState,
    _phase_collect_guards,
    _phase_collect_pre_edit_guards,
)
from invarlock.core.runner_runtime.guards import (
    _normalize_guard_result,
    guard_phase,
    prepare_guards_phase,
)
from invarlock.core.types import GuardValidationResult
from invarlock.guards.rmt import RMTGuard
from invarlock.guards.spectral import SpectralGuard
from invarlock.reporting.guards_invariants import _extract_invariants
from invarlock.reporting.run_report_payloads import build_guard_entries


class _VectorOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vector = nn.Parameter(torch.ones(4))


def _baseline_spectral_evidence(model: nn.Module) -> dict[str, Any]:
    guard = SpectralGuard(
        correction_enabled=False,
        degeneracy={"enabled": False},
    )
    guard.prepare(model, None, None, {})
    return {
        "name": "spectral",
        **_normalize_guard_result(guard.validate(model, None, {})),
    }


class _GuardRunner:
    def _log_event(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def _resolve_policy_flags(self, _config: Any) -> dict[str, bool]:
        return {"strict_guard_prepare": True}

    def _resolve_guard_policies(
        self, _report: RunReport, _auto_config: dict[str, Any] | None
    ) -> dict[str, dict[str, Any]]:
        return {}


def test_paired_spectral_guard_monitors_in_budget_external_family_cap() -> None:
    baseline = nn.Sequential(nn.Linear(4, 4, bias=False))
    subject = nn.Sequential(nn.Linear(4, 4, bias=False))
    with torch.no_grad():
        baseline[0].weight.copy_(torch.eye(4))
        subject[0].weight.copy_(10.0 * torch.eye(4))

    evidence = _baseline_spectral_evidence(baseline)
    guard = SpectralGuard(
        correction_enabled=False,
        degeneracy={"enabled": False},
    )
    report = RunReport(
        context={
            "profile": "release",
            "baseline_guard_evidence_required": True,
            "baseline_guard_evidence": {"spectral": evidence},
        }
    )
    runner = _GuardRunner()
    prepare_guards_phase(
        runner,
        subject,
        None,
        [guard],
        None,
        report,
        config=RunConfig(),
    )
    results = guard_phase(runner, subject, None, [guard], report)
    result = results["spectral"]

    assert report.meta["baseline_guard_evidence"]["spectral"]["ready"] is True
    assert result["passed"] is True
    assert result["decision"] == "monitor"
    assert result["metrics"]["caps_applied"] == 1
    assert result["metrics"]["caps_exceeded"] is False
    assert result["metrics"]["corrections_attempted"] == 0
    assert result["metrics"]["corrections_applied"] == 0
    assert result["correction_ledger"]["policy_result"] == "correction_disabled"
    assert result["correction_ledger"]["corrections"][0]["attempted"] is False
    assert result["metrics"]["baseline_source"] == "external_run"
    assert result["metrics"]["modules_checked"] == 1
    assert result["baseline_metrics"]["module_sigmas"]["0"] < 2.0
    assert result["final_metrics"]["0"] > 9.0


def test_spectral_zero_measurement_is_assurance_blocking() -> None:
    model = _VectorOnlyModel()
    guard = SpectralGuard(correction_enabled=False)
    guard.prepare(model, None, None, {})

    result = guard.validate(model, None, {})

    assert result.passed is False
    assert result.decision == "block"
    assert result.metrics["modules_checked"] == 0
    assert result.extras["supported"] is False
    assert result.extras["reason"] == "no_eligible_modules_measured"
    assert result.extras["assurance_blocking"] is True
    assert result.extras["status"] == "unsupported"
    assert result.extras["measurement_inventory"]["validate"]["measured_modules"] == []


def test_spectral_baseline_report_entry_preserves_decision_measurements() -> None:
    model = nn.Sequential(nn.Linear(3, 3, bias=False))
    evidence = _baseline_spectral_evidence(model)

    entries = build_guard_entries({"spectral": evidence})

    assert len(entries) == 1
    assert entries[0]["baseline_metrics"]["module_sigmas"]
    assert entries[0]["final_metrics"]
    assert "final_degeneracy" in entries[0]
    assert entries[0]["module_family_map"]


def test_paired_rmt_guard_uses_external_family_and_module_measurements() -> None:
    guard = RMTGuard(epsilon_default=0.01)
    measurement = {
        "edge_risk_by_family": {"ffn": 10.0},
        "edge_risk_by_module": {"mlp": 10.0},
    }
    guard._compute_activation_edge_risk = lambda *_args, **_kwargs: dict(measurement)
    contract = {
        "kind": "activation_edge_risk",
        "estimator": dict(guard.estimator),
        "activation_sampling": dict(guard.activation_sampling),
    }
    evidence = {
        "name": "rmt",
        "metrics": {
            "edge_risk_by_family": {"ffn": 1.0},
            "edge_risk_by_module": {"mlp": 1.0},
            "measurement_contract": contract,
        },
    }
    report = RunReport(
        context={
            "profile": "release",
            "baseline_guard_evidence_required": True,
            "baseline_guard_evidence": {"rmt": evidence},
        }
    )
    guard.set_run_context(report)
    guard.prepare(nn.Linear(2, 2, bias=False), None, [object()], {})

    load_result = guard.load_external_baseline_evidence()
    result = guard.validate(nn.Linear(2, 2, bias=False), None, report.context)

    assert load_result["ready"] is True
    assert result.passed is False
    assert result.decision == "block"
    assert result.metrics["baseline_source"] == "external_run"
    assert result.metrics["edge_risk_by_family_base"] == {"ffn": 1.0}
    assert result.metrics["edge_risk_by_family"] == {"ffn": 10.0}


class _StageGuard(Guard):
    def __init__(self, name: str, passed: bool) -> None:
        self.name = name
        self._passed = passed
        self.calls: list[str] = []

    def validate(
        self, model: Any, adapter: Any, context: dict[str, Any]
    ) -> GuardValidationResult:
        _ = adapter, context
        self.calls.append(str(model.stage))
        return GuardValidationResult(
            passed=self._passed,
            decision="allow" if self._passed else "block",
            metrics={"stage_seen": str(model.stage)},
        )


class _StageRunner:
    def _log_event(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def _guard_phase(self, *args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        return guard_phase(self, *args, **kwargs)


@dataclass
class _StageModel:
    stage: str = "pre"


def test_canonical_invariant_stages_are_executed_and_preserved_independently() -> None:
    model = _StageModel()
    pre = _StageGuard("invariants", passed=True)
    spectral = _StageGuard("spectral", passed=True)
    rmt = _StageGuard("rmt", passed=True)
    variance = _StageGuard("variance", passed=True)
    post = _StageGuard("invariants", passed=True)
    request = RunnerExecutionRequest(
        model=model,
        adapter=object(),
        edit=object(),
        guards=[pre, spectral, rmt, variance, post],
        config=RunConfig(),
    )
    state = RunnerExecutionState(
        request=request,
        report=RunReport(),
        timings={},
        guard_timings={},
        memory_snapshots=[],
        edit_runtime=EditRuntime(),
    )
    runner = _StageRunner()

    _phase_collect_pre_edit_guards(runner, state)
    model.stage = "post"
    _phase_collect_guards(runner, state)

    assert pre.calls == ["pre"]
    assert post.calls == ["post"]
    assert state.guard_results is not None
    assert state.guard_results["invariants"]["passed"] is True
    assert state.guard_results["invariants"]["stage"] == "pre"
    assert state.guard_results["invariants_post"]["passed"] is True
    assert state.guard_results["invariants_post"]["stage"] == "post"


def test_failed_pre_edit_invariant_gate_aborts_before_edit() -> None:
    model = _StageModel()
    pre = _StageGuard("invariants", passed=False)
    request = RunnerExecutionRequest(
        model=model,
        adapter=object(),
        edit=object(),
        guards=[
            pre,
            _StageGuard("spectral", passed=True),
            _StageGuard("rmt", passed=True),
            _StageGuard("variance", passed=True),
            _StageGuard("invariants", passed=True),
        ],
        config=RunConfig(),
    )
    state = RunnerExecutionState(
        request=request,
        report=RunReport(),
        timings={},
        guard_timings={},
        memory_snapshots=[],
        edit_runtime=EditRuntime(),
    )

    with pytest.raises(RuntimeError, match="edit was not executed"):
        _phase_collect_pre_edit_guards(_StageRunner(), state)

    assert model.stage == "pre"
    assert state.report.meta["pre_edit_guard_failures"] == ["invariants"]


def test_report_invariants_fail_when_pre_stage_failed_but_post_stage_passed() -> None:
    result = _extract_invariants(
        {
            "metrics": {},
            "guards": [
                {
                    "name": "invariants",
                    "stage": "pre",
                    "passed": False,
                    "decision": "block",
                    "metrics": {"checks_performed": 1, "fatal_violations": 1},
                    "violations": [
                        {
                            "type": "non_finite_tensor",
                            "severity": "fatal",
                            "message": "pre failed",
                        }
                    ],
                    "details": {},
                },
                {
                    "name": "invariants_post",
                    "stage": "post",
                    "passed": True,
                    "decision": "allow",
                    "metrics": {"checks_performed": 1, "fatal_violations": 0},
                    "violations": [],
                    "details": {},
                },
            ],
        }
    )

    assert result["pre"] == "fail"
    assert result["post"] == "pass"
    assert result["status"] == "fail"
    assert result["passed"] is False
    assert result["decision"] == "block"
