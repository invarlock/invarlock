"""Typed run orchestration owner for config-driven run commands."""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from invarlock.core import run_orchestrator_execute as run_orchestrator_execute_mod
from invarlock.core.exceptions import ConfigError, InvarlockError
from invarlock.core.run_execution_context_policy import (
    build_run_context_payload as _build_run_context_payload_impl,
)
from invarlock.core.run_execution_context_policy import (
    build_run_execution_config_payloads as _build_run_execution_config_payloads_impl,
)
from invarlock.core.run_orchestrator_types import (
    RunAdapterSelectedEvent,
    RunAttemptStartedEvent,
    RunAutoTuneAdjustmentEvent,
    RunBaselineScheduleLoadedEvent,
    RunCalibrationBatchSizesDebugEvent,
    RunCleanupStatusEvent,
    RunConfigLoadedEvent,
    RunConfigLoadingEvent,
    RunDatasetLoadingEvent,
    RunDeterministicSeedsEvent,
    RunDeviceResolvedEvent,
    RunDiagnosticEvent,
    RunEditSelectedEvent,
    RunEvaluationReportFailedEvent,
    RunEvaluationReportPassedEvent,
    RunEvaluationReportStartedEvent,
    RunExecutePipelineEvent,
    RunExecutionEvent,
    RunExecutionFailure,
    RunExecutionObserver,
    RunExecutionOutcome,
    RunExecutionRequest,
    RunExecutionResult,
    RunExecutionServices,
    RunFailureEvent,
    RunGuardChainResolvedEvent,
    RunGuardOverheadSummaryEvent,
    RunLoadModelOnceEvent,
    RunMaskedTokensDebugEvent,
    RunOutputDirectoryReadyEvent,
    RunPipelineStartedEvent,
    RunPreviewLabelsDebugEvent,
    RunPrimaryMetricSummaryEvent,
    RunRetryAttemptStartedEvent,
    RunRetryExhaustedEvent,
    RunRetrySummaryEvent,
    RunRetryValidationErrorEvent,
    RunSnapshotModeEvent,
    RunTelemetryFailedEvent,
    RunTelemetrySavedEvent,
    TimingSummaryPayload,
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
    build_restore_failure_attempt_summary as _build_restore_failure_attempt_summary_impl,
)
from invarlock.core.run_retry_policy import (
    decide_failed_retry_transition as _decide_failed_retry_transition_impl,
)
from invarlock.core.run_retry_policy import (
    record_retry_attempt as _record_retry_attempt_impl,
)
from invarlock.core.run_retry_policy import (
    resolve_retry_validation_transition as _resolve_retry_validation_transition_impl,
)
from invarlock.core.run_timing_policy import (
    build_timing_summary_payload as _build_timing_summary_payload_impl,
)
from invarlock.model_utils import set_seed

# class RunExecutionEvent
# class RunLifecycleEvent
# class RunDiagnosticEvent
# class RunContextEvent
# class RunAggregateEvent
# class RunFailureEvent
# class RunPrimaryMetricSummaryEvent
# Typed contracts live in `invarlock.core.run_orchestrator_types` and are
# re-exported here so the owner boundary remains stable.

__all__ = [
    "RunAdapterSelectedEvent",
    "RunAttemptStartedEvent",
    "RunAutoTuneAdjustmentEvent",
    "RunBaselineScheduleLoadedEvent",
    "RunCalibrationBatchSizesDebugEvent",
    "RunCleanupStatusEvent",
    "RunConfigLoadedEvent",
    "RunConfigLoadingEvent",
    "RunDatasetLoadingEvent",
    "RunDeterministicSeedsEvent",
    "RunDeviceResolvedEvent",
    "RunDiagnosticEvent",
    "RunEditSelectedEvent",
    "RunEvaluationReportFailedEvent",
    "RunEvaluationReportPassedEvent",
    "RunEvaluationReportStartedEvent",
    "RunExecutePipelineEvent",
    "RunExecutionEvent",
    "RunExecutionFailure",
    "RunExecutionObserver",
    "RunExecutionOutcome",
    "RunExecutionRequest",
    "RunExecutionResult",
    "RunExecutionServices",
    "RunFailureEvent",
    "RunGuardChainResolvedEvent",
    "RunGuardOverheadSummaryEvent",
    "RunLoadModelOnceEvent",
    "RunMaskedTokensDebugEvent",
    "RunOutputDirectoryReadyEvent",
    "RunPipelineStartedEvent",
    "RunPreviewLabelsDebugEvent",
    "RunPrimaryMetricSummaryEvent",
    "RunRetryAttemptStartedEvent",
    "RunRetryExhaustedEvent",
    "RunRetrySummaryEvent",
    "RunRetryValidationErrorEvent",
    "RunSnapshotModeEvent",
    "RunTelemetryFailedEvent",
    "RunTelemetrySavedEvent",
    "TimingSummaryPayload",
    "_RunExecutionHalt",
    "execute_run_request",
]


def _coerce_float(value: Any, default: float) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return float(default)
    return coerced if math.isfinite(coerced) else float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _sync_execute_globals() -> None:
    synced_globals = {
        "_build_restore_failure_attempt_summary_impl": (
            _build_restore_failure_attempt_summary_impl
        ),
        "_build_run_context_payload_impl": _build_run_context_payload_impl,
        "_build_run_execution_config_payloads_impl": (
            _build_run_execution_config_payloads_impl
        ),
        "_build_timing_summary_payload_impl": _build_timing_summary_payload_impl,
        "_coerce_float": _coerce_float,
        "_coerce_int": _coerce_int,
        "_decide_failed_retry_transition_impl": _decide_failed_retry_transition_impl,
        "_record_retry_attempt_impl": _record_retry_attempt_impl,
        "_resolve_guard_overhead_threshold_impl": (
            _resolve_guard_overhead_threshold_impl
        ),
        "_resolve_pm_acceptance_range_impl": _resolve_pm_acceptance_range_impl,
        "_resolve_pm_drift_band_impl": _resolve_pm_drift_band_impl,
        "_resolve_retry_validation_transition_impl": (
            _resolve_retry_validation_transition_impl
        ),
        "_should_measure_overhead_impl": _should_measure_overhead_impl,
        "ConfigError": ConfigError,
        "InvarlockError": InvarlockError,
        "Path": Path,
        "cast": cast,
        "contextmanager": contextmanager,
        "datetime": datetime,
        "math": math,
        "np": np,
        "os": os,
        "perf_counter": perf_counter,
        "set_seed": set_seed,
    }
    for name, value in synced_globals.items():
        setattr(run_orchestrator_execute_mod, name, value)


def execute_run_request(
    request: RunExecutionRequest,
    *,
    services: RunExecutionServices,
    observer: RunExecutionObserver | None = None,
) -> RunExecutionOutcome:
    _sync_execute_globals()
    return run_orchestrator_execute_mod.execute_run_request_impl(
        request,
        services=services,
        observer=observer,
    )
