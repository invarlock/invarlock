"""Internal execution implementation for config-driven run orchestration."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import partial
from time import perf_counter
from typing import Any, NoReturn

import numpy as np  # noqa: F401

import invarlock.core.run_orchestrator_execute_helpers as _execute_helpers
from invarlock.core.exceptions import InvarlockError
from invarlock.core.run_execution_context_policy import (
    build_run_context_payload as _build_run_context_payload_impl,
)
from invarlock.core.run_execution_context_policy import (
    build_run_execution_config_payloads as _build_run_execution_config_payloads_impl,
)
from invarlock.core.run_orchestrator_execute_helpers import (
    RunEventEmitter as RunEventEmitter,
)
from invarlock.core.run_orchestrator_execute_helpers import (
    _cfg_section_value,
)
from invarlock.core.run_orchestrator_execute_helpers import (
    _coerce_float as _coerce_float,
)
from invarlock.core.run_orchestrator_execute_helpers import (
    _coerce_int as _coerce_int,
)
from invarlock.core.run_orchestrator_execute_pipeline import (
    _execute_run_pipeline_steps,
)
from invarlock.core.run_orchestrator_types import (
    RunCleanupStatusEvent,
    RunDiagnosticEvent,
    RunExecutionEvent,
    RunExecutionFailure,
    RunExecutionObserver,
    RunExecutionOutcome,
    RunExecutionRequest,
    RunExecutionResult,
    RunExecutionServices,
    RunFailureEvent,
    RunGuardOverheadSummaryEvent,
    RunRetrySummaryEvent,
    _RunExecutionHalt,
)
from invarlock.core.run_policy import (
    resolve_guard_overhead_threshold as _resolve_guard_overhead_threshold_impl,
)
from invarlock.core.run_policy import (
    resolve_pm_acceptance_range as _resolve_pm_acceptance_range_impl,
)
from invarlock.core.run_policy import (
    resolve_pm_drift_band as _resolve_pm_drift_band_impl,
)
from invarlock.core.run_policy import (
    should_measure_overhead as _should_measure_overhead_impl,
)
from invarlock.core.run_retry_policy import (
    resolve_retry_validation_transition as _resolve_retry_validation_transition_impl,
)
from invarlock.core.run_timing_policy import (
    build_timing_summary_payload as _build_timing_summary_payload_impl,
)

# class RunExecutionEvent
# class RunLifecycleEvent
# class RunDiagnosticEvent
# class RunContextEvent
# class RunAggregateEvent
# class RunFailureEvent
# class RunPrimaryMetricSummaryEvent
# Typed contracts live in `invarlock.core.run_orchestrator_types` and are
# re-exported here so the owner boundary remains stable.


def _emit_run_diagnostic(
    emit: RunEventEmitter,
    *,
    origin: str | None = None,
    code: str | None = None,
    summary: str | None = None,
    level: str | None = None,
    **context: Any,
) -> None:
    emit(
        RunDiagnosticEvent(
            source=origin,
            code=code,
            summary=summary,
            level=level,
            context=dict(context),
        )
    )


def _emit_run_guard_overhead_summary(
    emit: RunEventEmitter,
    guard_overhead_info: dict[str, Any],
    *,
    default_threshold: float,
) -> None:
    emit(
        RunGuardOverheadSummaryEvent(
            guard_overhead_info=guard_overhead_info,
            default_threshold=default_threshold,
        )
    )


def _emit_run_retry_summary(
    emit: RunEventEmitter,
    retry_controller: Any | None,
) -> None:
    if not retry_controller or not getattr(retry_controller, "attempt_history", None):
        return
    try:
        summary = retry_controller.get_attempt_summary()
    except (AttributeError, KeyError, TypeError, ValueError):
        return
    if not isinstance(summary, dict):
        return
    emit(RunRetrySummaryEvent(summary=summary))


def _raise_run_halt(
    emit: RunEventEmitter,
    halt_error_type: type[_RunExecutionHalt],
    *,
    code: str,
    summary: str | None = None,
    error: Exception | None = None,
    **context: Any,
) -> NoReturn:
    failure = RunExecutionFailure(
        code=code,
        summary=summary,
        error=error,
        context=dict(context),
    )
    emit(RunFailureEvent(failure=failure))
    raise halt_error_type(failure)


def _emit_transition_diagnostic(
    emit_diagnostic: Any,
    source: str,
    diagnostic: Any,
) -> None:
    code = getattr(diagnostic, "code", None)
    if isinstance(code, str) and code:
        details = getattr(diagnostic, "details", None)
        context = getattr(diagnostic, "context", None)
        payload = {}
        if isinstance(details, dict):
            payload.update(details)
        if isinstance(context, dict):
            payload.update(context)
        payload.setdefault("diagnostic_source", source)
        summary = getattr(diagnostic, "summary", None)
        if not isinstance(summary, str) or not summary:
            message = getattr(diagnostic, "message", None)
            summary = message if isinstance(message, str) and message else None
        emit_diagnostic(origin=source, code=code, summary=summary, **payload)
        return
    kind = getattr(diagnostic, "kind", None)
    if isinstance(kind, str) and kind:
        payload = {}
        metadata = getattr(diagnostic, "metadata", None)
        if isinstance(metadata, dict):
            payload.update(metadata)
        context = getattr(diagnostic, "context", None)
        if isinstance(context, dict):
            payload.update(context)
        payload.setdefault("diagnostic_source", source)
        level = getattr(diagnostic, "level", None)
        if not isinstance(level, str) or not level:
            severity = getattr(diagnostic, "severity", None)
            level = severity if isinstance(severity, str) and severity else None
        summary = getattr(diagnostic, "summary", None)
        if not isinstance(summary, str) or not summary:
            message = getattr(diagnostic, "message", None)
            summary = message if isinstance(message, str) and message else None
        emit_diagnostic(
            origin=source,
            code=kind,
            summary=summary,
            level=level,
            **payload,
        )
        return
    payload = {"diagnostic_source": source}
    metadata = getattr(diagnostic, "metadata", None)
    if isinstance(metadata, dict):
        payload.update(metadata)
    details = getattr(diagnostic, "details", None)
    if isinstance(details, dict):
        payload.update(details)
    context = getattr(diagnostic, "context", None)
    if isinstance(context, dict):
        payload.update(context)
    level = getattr(diagnostic, "level", None)
    if not isinstance(level, str) or not level:
        severity = getattr(diagnostic, "severity", None)
        level = severity if isinstance(severity, str) and severity else None
    summary = getattr(diagnostic, "summary", None)
    if not isinstance(summary, str) or not summary:
        message = getattr(diagnostic, "message", None)
        summary = message if isinstance(message, str) and message else None
    if len(payload) > 1 or (isinstance(summary, str) and summary):
        emit_diagnostic(
            origin=source,
            code="transition_diagnostic",
            summary=summary,
            level=level,
            **payload,
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
    elif isinstance(error, ModuleNotFoundError | ImportError) and "torch" in str(error):
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


def execute_run_request_impl(
    request: RunExecutionRequest,
    *,
    services: RunExecutionServices,
    observer: RunExecutionObserver | None = None,
) -> RunExecutionOutcome:
    """Execute a config-driven run and return a typed outcome/event stream."""

    config_value_exceptions = (AttributeError, TypeError, ValueError, KeyError)
    numeric_exceptions = (TypeError, ValueError, OverflowError)
    optional_runtime_exceptions = (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    )

    profile_normalized = (str(request.profile or "")).strip().lower()
    timings: dict[str, float] = {}
    collect_timings = bool(request.capture_timings or request.telemetry)
    total_start: float | None = perf_counter() if collect_timings else None
    snapshot_tmpdir: Any | None = None
    outcome_result: RunExecutionResult | None = None
    outcome_failure: RunExecutionFailure | None = None
    emitted_events: list[RunExecutionEvent] = []

    def _emit(event: RunExecutionEvent) -> None:
        emitted_events.append(event)
        if observer is not None:
            observer(event)

    _emit_diagnostic = partial(_emit_run_diagnostic, _emit)
    _emit_guard_overhead_summary = partial(_emit_run_guard_overhead_summary, _emit)
    _emit_retry_summary = partial(_emit_run_retry_summary, _emit)

    def _halt(
        code: str,
        *,
        summary: str | None = None,
        error: Exception | None = None,
        **context: Any,
    ) -> NoReturn:
        _raise_run_halt(
            _emit,
            _RunExecutionHalt,
            code=code,
            summary=summary,
            error=error,
            **context,
        )

    @contextmanager
    def _record_timed_step(key: str):
        start = perf_counter()
        yield
        elapsed = max(0.0, float(perf_counter() - start))
        if collect_timings:
            timings[key] = elapsed

    def _fail_run(message: str, *, error: Exception | None = None) -> None:
        _halt("pipeline_failed", summary=message, error=error)

    _emit_transition = partial(_emit_transition_diagnostic, _emit_diagnostic)
    _cfg_value = partial(
        _cfg_section_value,
        config_value_exceptions=config_value_exceptions,
    )

    optional_dep_unset = object()
    optional_torch_cache = optional_dep_unset

    def _optional_torch() -> Any | None:
        nonlocal optional_torch_cache
        if optional_torch_cache is optional_dep_unset:
            loaded = services.get_torch()
            optional_torch_cache = loaded if loaded else None
        return optional_torch_cache

    def _require_torch() -> Any:
        loaded = _optional_torch()
        if loaded is not None:
            return loaded
        _halt("torch_missing")
        raise AssertionError("unreachable after torch_missing halt")  # pragma: no cover

    try:
        _execute_helpers._build_run_context_payload_impl = (
            _build_run_context_payload_impl
        )
        _execute_helpers._build_run_execution_config_payloads_impl = (
            _build_run_execution_config_payloads_impl
        )
        _execute_helpers._resolve_pm_acceptance_range_impl = (
            _resolve_pm_acceptance_range_impl
        )
        _execute_helpers._resolve_pm_drift_band_impl = _resolve_pm_drift_band_impl
        _execute_helpers._resolve_guard_overhead_threshold_impl = (
            _resolve_guard_overhead_threshold_impl
        )
        _execute_helpers._should_measure_overhead_impl = _should_measure_overhead_impl
        outcome_result, snapshot_tmpdir = _execute_run_pipeline_steps(
            request=request,
            services=services,
            profile_normalized=profile_normalized,
            collect_timings=collect_timings,
            total_start=total_start,
            timings=timings,
            emit=_emit,
            emit_diagnostic=_emit_diagnostic,
            emit_guard_overhead_summary=_emit_guard_overhead_summary,
            emit_retry_summary=_emit_retry_summary,
            emit_transition=_emit_transition,
            halt=_halt,
            fail_run=_fail_run,
            record_timed_step=_record_timed_step,
            optional_torch=_optional_torch,
            require_torch=_require_torch,
            build_timing_summary_payload_fn=_build_timing_summary_payload_impl,
            resolve_retry_validation_transition_fn=(
                _resolve_retry_validation_transition_impl
            ),
            cfg_value=_cfg_value,
            config_value_exceptions=config_value_exceptions,
            numeric_exceptions=numeric_exceptions,
            optional_runtime_exceptions=optional_runtime_exceptions,
        )
    except FileNotFoundError as error:
        outcome_failure = RunExecutionFailure(
            code="config_file_missing",
            summary=str(error),
            error=error,
            context={"path": str(error)},
        )
        _emit(RunFailureEvent(failure=outcome_failure))
    except InvarlockError as error:
        outcome_failure = RunExecutionFailure(
            code="invarlock_error",
            summary=str(error),
            error=error,
        )
        _emit(RunFailureEvent(failure=outcome_failure))
    except _RunExecutionHalt as halt:
        outcome_failure = halt.failure
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
        MemoryError,
        ImportError,
        ModuleNotFoundError,
    ) as error:
        outcome_failure = _map_pipeline_failure(error, emit=_emit)
    finally:
        _cleanup_snapshot_tmpdir(
            snapshot_tmpdir=snapshot_tmpdir,
            no_cleanup=bool(request.no_cleanup),
            emit=_emit,
        )
    return RunExecutionOutcome(
        ok=outcome_failure is None and outcome_result is not None,
        result=outcome_result,
        failure=outcome_failure,
        events=tuple(emitted_events),
    )
