"""Typed execution plan for the core guarded-run pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from invarlock.observability.metrics import (
    capture_memory_snapshot,
    reset_peak_memory_stats,
    summarize_memory_snapshots,
)

from .api import (
    EditLike,
    EditRuntime,
    Guard,
    ModelAdapter,
    ModelEdit,
    RunConfig,
    RunReport,
)
from .types import LogLevel, RunStatus


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


def _phase_collect_guards(runner: Any, state: RunnerExecutionState) -> None:
    request = state.request
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
    RunnerPhase("edit", _phase_apply_edit),
    RunnerPhase("guards", _phase_collect_guards),
    RunnerPhase("eval", _phase_evaluate),
    RunnerPhase("finalize", _phase_finalize),
)


def _run_timed_phase(
    runner: Any,
    state: RunnerExecutionState,
    phase: RunnerPhase,
    *,
    record_timing_fn: Any,
    capture_memory_fn: Any,
) -> None:
    reset_peak_memory_stats()
    phase_start = time.perf_counter()
    try:
        phase.action(runner, state)
    finally:
        record_timing_fn(state.timings, phase.key, phase_start)
        capture_memory_fn(
            state.memory_snapshots,
            phase.key,
            capture_fn=capture_memory_snapshot,
        )


def build_runner_execution_state(
    request: RunnerExecutionRequest,
    *,
    report: RunReport,
    profile_from_context_fn: Any,
) -> RunnerExecutionState:
    edit_runtime = request.edit_runtime
    if edit_runtime is None:
        edit_runtime = EditRuntime(
            profile=profile_from_context_fn(request.config.context),
            verbose=bool(request.config.verbose),
        )
    return RunnerExecutionState(
        request=request,
        report=report,
        timings={},
        guard_timings={},
        memory_snapshots=[],
        edit_runtime=edit_runtime,
    )


def execute_runner_execution_plan(
    runner: Any,
    request: RunnerExecutionRequest,
    *,
    initialize_run_report_fn: Any,
    collect_cuda_flags_fn: Any,
    profile_from_context_fn: Any,
    record_timing_fn: Any,
    capture_memory_fn: Any,
    finalize_run_report_fn: Any,
    merge_execution_metrics_fn: Any,
    runner_execution_errors: tuple[type[BaseException], ...],
) -> RunReport:
    runner._initialize_services(request.config)
    runner._active_model = request.model
    runner._active_adapter = request.adapter

    report = cast(
        RunReport,
        initialize_run_report_fn(
            config=request.config,
            serialized_config=runner._serialize_config(request.config),
            cuda_flags=collect_cuda_flags_fn(),
            auto_config=request.auto_config,
            report_factory=RunReport,
        ),
    )
    report.status = RunStatus.RUNNING.value
    state = build_runner_execution_state(
        request,
        report=report,
        profile_from_context_fn=profile_from_context_fn,
    )
    total_start = time.perf_counter()

    try:
        runner._log_event(
            "runner",
            "start",
            LogLevel.INFO,
            {
                "edit": request.edit.name,
                "guards": [guard.name for guard in request.guards],
                "context": report.context,
            },
        )
        for phase in RUNNER_PHASES:
            _run_timed_phase(
                runner,
                state,
                phase,
                record_timing_fn=record_timing_fn,
                capture_memory_fn=capture_memory_fn,
            )

        final_status = _require_phase_value(state.final_status, phase="complete")
        finalize_run_report_fn(report, final_status=final_status)
        runner._log_event(
            "runner",
            "complete",
            LogLevel.INFO,
            {"status": final_status, "duration": report.meta["duration"]},
        )
        return report
    except runner_execution_errors as error:
        runner._handle_error(
            error, report, model=request.model, adapter=request.adapter
        )
        return report
    finally:
        record_timing_fn(state.timings, "total", total_start)
        merge_execution_metrics_fn(
            report,
            timings=state.timings,
            guard_timings=state.guard_timings,
            memory_snapshots=state.memory_snapshots,
            memory_summary=summarize_memory_snapshots(state.memory_snapshots)
            if state.memory_snapshots
            else {},
        )
        runner._active_model = None
        runner._active_adapter = None
        runner._cleanup_services()


__all__ = [
    "RUNNER_PHASES",
    "RunnerExecutionRequest",
    "RunnerExecutionState",
    "RunnerPhase",
    "build_runner_execution_state",
    "execute_runner_execution_plan",
]
