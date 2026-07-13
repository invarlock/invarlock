"""Typed execution plan for the core guarded-run pipeline."""

from __future__ import annotations

import time
from typing import Any, cast

from invarlock.observability.metrics import (
    capture_memory_snapshot,
    reset_peak_memory_stats,
    summarize_memory_snapshots,
)

from ..api import EditRuntime, RunReport
from ..types import LogLevel, RunStatus
from . import execution_phases as _runner_execution_phases
from .execution_phases import (
    RunnerExecutionRequest,
    RunnerExecutionState,
    RunnerPhase,
)

_require_phase_value = _runner_execution_phases._require_phase_value
_phase_prepare_model = _runner_execution_phases._phase_prepare_model
_phase_prepare_guards = _runner_execution_phases._phase_prepare_guards
_phase_apply_edit = _runner_execution_phases._phase_apply_edit
_uses_canonical_staged_guards = _runner_execution_phases._uses_canonical_staged_guards
_phase_collect_pre_edit_guards = _runner_execution_phases._phase_collect_pre_edit_guards
_phase_collect_guards = _runner_execution_phases._phase_collect_guards
_phase_evaluate = _runner_execution_phases._phase_evaluate
_phase_finalize = _runner_execution_phases._phase_finalize
RUNNER_PHASES = _runner_execution_phases.RUNNER_PHASES


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
