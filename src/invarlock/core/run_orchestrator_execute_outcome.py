"""Outcome and failure handling for config-driven run orchestration."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

from invarlock.core.run_orchestrator_execute_helpers import RunEventEmitter
from invarlock.core.run_orchestrator_types import (
    RunCleanupStatusEvent,
    RunExecutionFailure,
    RunExecutionResult,
    RunFailureEvent,
    TimingSummaryPayload,
)
from invarlock.core.run_timing_policy import build_timing_summary_payload


def _build_outcome_result(
    *,
    capture_timings: bool,
    total_start: float | None,
    timings: dict[str, float],
    report: dict[str, Any],
    report_path_out: str | None,
    build_timing_summary_payload_fn: Any = build_timing_summary_payload,
) -> RunExecutionResult:
    timing_summary: TimingSummaryPayload | None = None
    if capture_timings:
        total_duration = (
            max(0.0, float(perf_counter() - total_start))
            if total_start is not None
            else None
        )
        summary_payload = build_timing_summary_payload_fn(
            timings=timings,
            total_duration=total_duration,
            report=report if isinstance(report, dict) else None,
        )
        if summary_payload is not None:
            timings = dict(summary_payload.timings)
            timing_summary = summary_payload
    return RunExecutionResult(
        report_path=report_path_out,
        timings=dict(timings),
        timing_summary=timing_summary,
    )


def _map_pipeline_failure(
    error: BaseException,
    *,
    emit: RunEventEmitter,
) -> RunExecutionFailure:
    error_obj = error if isinstance(error, Exception) else Exception(str(error))
    if os.environ.get("INVARLOCK_DEBUG_TRACE"):
        import traceback

        traceback.print_exc()
    if isinstance(error, ValueError) and "Invalid RunReport" in str(error):
        failure = RunExecutionFailure(
            code="schema_invalid_run_report",
            summary=str(error),
            error=error_obj,
        )
    elif isinstance(error, (ModuleNotFoundError, ImportError)) and "torch" in str(
        error
    ):
        failure = RunExecutionFailure(
            code="torch_missing",
            summary=str(error),
            error=error_obj,
        )
    else:
        failure = RunExecutionFailure(
            code="pipeline_failed",
            summary=str(error),
            error=error_obj,
        )
    emit(RunFailureEvent(failure=failure))
    return failure


def _cleanup_snapshot_tmpdir(
    *,
    snapshot_tmpdir: Any | None,
    no_cleanup: bool,
    emit: RunEventEmitter,
) -> None:
    try:
        if snapshot_tmpdir and not no_cleanup:
            try:
                import shutil as _sh

                _sh.rmtree(snapshot_tmpdir, ignore_errors=True)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
            finally:
                emit(RunCleanupStatusEvent(removed=True))
        else:
            emit(RunCleanupStatusEvent(removed=False))
    except (AttributeError, NameError, TypeError, OSError):
        return
