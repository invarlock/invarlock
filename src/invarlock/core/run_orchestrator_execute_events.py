"""Event emission helpers for config-driven run orchestration."""

from __future__ import annotations

from typing import Any, NoReturn

from invarlock.core.run_orchestrator_execute_helpers import RunEventEmitter
from invarlock.core.run_orchestrator_types import (
    RunDiagnosticEvent,
    RunExecutionFailure,
    RunFailureEvent,
    RunGuardOverheadSummaryEvent,
    RunRetrySummaryEvent,
    _RunExecutionHalt,
)


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
