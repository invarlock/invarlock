"""State models and phase actions for the guarded runner pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..api import (
    EditLike,
    EditRuntime,
    Guard,
    ModelAdapter,
    ModelEdit,
    RunConfig,
    RunReport,
)
from ..assurance_contract import CANONICAL_GUARD_CHAIN


@dataclass(frozen=True)
class RunnerExecutionRequest:
    model: Any
    adapter: ModelAdapter
    edit: ModelEdit | EditLike
    guards: list[Guard]
    config: RunConfig
    calibration_data: Any = None
    auto_config: dict[str, Any] | None = None
    edit_config: dict[str, Any] | None = None
    edit_runtime: EditRuntime | None = None
    preview_n: int | None = None
    final_n: int | None = None


@dataclass
class RunnerExecutionState:
    request: RunnerExecutionRequest
    report: RunReport
    timings: dict[str, float]
    guard_timings: dict[str, float]
    memory_snapshots: list[dict[str, Any]]
    edit_runtime: EditRuntime
    model_desc: dict[str, Any] | None = None
    guard_results: dict[str, dict[str, Any]] | None = None
    pre_guard_results: dict[str, dict[str, Any]] | None = None
    metrics: dict[str, Any] | None = None
    final_status: str | None = None


@dataclass(frozen=True)
class RunnerPhase:
    key: str
    action: Callable[[Any, RunnerExecutionState], None]


def _require_phase_value(value: Any, *, phase: str) -> Any:
    if value is None:
        raise AssertionError(f"Runner phase '{phase}' ran before its dependency")
    return value


def _phase_prepare_model(runner: Any, state: RunnerExecutionState) -> None:
    request = state.request
    state.model_desc = runner._prepare_phase(
        request.model,
        request.adapter,
        state.report,
    )


def _phase_prepare_guards(runner: Any, state: RunnerExecutionState) -> None:
    request = state.request
    runner._prepare_guards_phase(
        request.model,
        request.adapter,
        request.guards,
        request.calibration_data,
        state.report,
        request.auto_config,
        request.config,
    )


def _phase_apply_edit(runner: Any, state: RunnerExecutionState) -> None:
    request = state.request
    model_desc = _require_phase_value(state.model_desc, phase="edit")
    runner._edit_phase(
        request.model,
        request.adapter,
        request.edit,
        model_desc,
        state.report,
        request.edit_config,
        state.edit_runtime,
    )


def _uses_canonical_staged_guards(guards: list[Guard]) -> bool:
    return tuple(str(guard.name) for guard in guards) == CANONICAL_GUARD_CHAIN


def _phase_collect_pre_edit_guards(runner: Any, state: RunnerExecutionState) -> None:
    request = state.request
    if not _uses_canonical_staged_guards(request.guards):
        state.pre_guard_results = {}
        return
    state.pre_guard_results = runner._guard_phase(
        request.model,
        request.adapter,
        [request.guards[0]],
        state.report,
        guard_timings=state.guard_timings,
        result_keys=["invariants"],
        result_stages=["pre"],
    )
    failed = [
        name
        for name, result in state.pre_guard_results.items()
        if result.get("passed") is not True
        or result.get("decision") in {"block", "rollback"}
    ]
    if failed:
        state.report.meta["pre_edit_guard_failures"] = failed
        raise RuntimeError(
            "Pre-edit invariant gate failed; the edit was not executed: "
            + ", ".join(failed)
        )


def _phase_collect_guards(runner: Any, state: RunnerExecutionState) -> None:
    request = state.request
    if _uses_canonical_staged_guards(request.guards):
        post_results = runner._guard_phase(
            request.model,
            request.adapter,
            request.guards[1:],
            state.report,
            guard_timings=state.guard_timings,
            result_keys=["spectral", "rmt", "variance", "invariants_post"],
            result_stages=[None, None, None, "post"],
        )
        state.guard_results = {
            **(state.pre_guard_results or {}),
            **post_results,
        }
        state.report.guards = state.guard_results
        return
    state.guard_results = runner._guard_phase(
        request.model,
        request.adapter,
        request.guards,
        state.report,
        guard_timings=state.guard_timings,
    )


def _phase_evaluate(runner: Any, state: RunnerExecutionState) -> None:
    request = state.request
    state.metrics = runner._eval_phase(
        request.model,
        request.adapter,
        request.calibration_data,
        state.report,
        request.preview_n,
        request.final_n,
        request.config,
    )


def _phase_finalize(runner: Any, state: RunnerExecutionState) -> None:
    request = state.request
    guard_results = _require_phase_value(state.guard_results, phase="finalize")
    metrics = _require_phase_value(state.metrics, phase="finalize")
    state.final_status = runner._finalize_phase(
        request.model,
        request.adapter,
        guard_results,
        metrics,
        request.config,
        state.report,
    )


RUNNER_PHASES: tuple[RunnerPhase, ...] = (
    RunnerPhase("prepare", _phase_prepare_model),
    RunnerPhase("prepare_guards", _phase_prepare_guards),
    RunnerPhase("guards_pre", _phase_collect_pre_edit_guards),
    RunnerPhase("edit", _phase_apply_edit),
    RunnerPhase("guards", _phase_collect_guards),
    RunnerPhase("eval", _phase_evaluate),
    RunnerPhase("finalize", _phase_finalize),
)

__all__ = [
    "RUNNER_PHASES",
    "RunnerExecutionRequest",
    "RunnerExecutionState",
    "RunnerPhase",
]
