"""Facade for attempt execution helpers in config-driven run orchestration."""

from __future__ import annotations

from invarlock.core import run_orchestrator_execute_attempts_impl as _impl
from invarlock.core.run_orchestrator_execute_helpers import (
    RunEventEmitter,
    _AttemptDecision,
    _AttemptExecutionState,
    _RunExecutionState,
)

_emit_attempt_start = _impl._emit_attempt_start
_build_skipped_guard_overhead_payload = _impl._build_skipped_guard_overhead_payload
_execute_attempt_core = _impl._execute_attempt_core
_should_export_model = _impl._should_export_model
_resolve_export_model_dir = _impl._resolve_export_model_dir
_maybe_export_model_artifacts = _impl._maybe_export_model_artifacts
_emit_primary_metric_summary_from_report = (
    _impl._emit_primary_metric_summary_from_report
)
_enforce_guard_overhead_budget = _impl._enforce_guard_overhead_budget
_handle_retry_validation = _impl._handle_retry_validation
_process_attempt_result = _impl._process_attempt_result
_execute_attempt_loop = _impl._execute_attempt_loop

__all__ = [
    "RunEventEmitter",
    "_AttemptDecision",
    "_AttemptExecutionState",
    "_RunExecutionState",
    "_emit_attempt_start",
    "_build_skipped_guard_overhead_payload",
    "_execute_attempt_core",
    "_should_export_model",
    "_resolve_export_model_dir",
    "_maybe_export_model_artifacts",
    "_emit_primary_metric_summary_from_report",
    "_enforce_guard_overhead_budget",
    "_handle_retry_validation",
    "_process_attempt_result",
    "_execute_attempt_loop",
]
