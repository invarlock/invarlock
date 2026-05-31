"""Typed run orchestration public facade for config-driven run commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from invarlock.core.run_policy import RunExecutionRequest, TimingSummaryPayload


class RunExecutionEvent:
    """Marker base class for typed run-orchestration events."""


class RunLifecycleEvent(RunExecutionEvent):
    """Marker base class for lifecycle/progress events."""


@dataclass(frozen=True)
class RunDiagnosticEvent(RunExecutionEvent):
    """Structured diagnostic emitted by the owner layer."""

    source: str | None = None
    code: str | None = None
    summary: str | None = None
    level: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


class RunContextEvent(RunExecutionEvent):
    """Marker base class for context events."""


class RunAggregateEvent(RunExecutionEvent):
    """Marker base class for aggregate/summary events."""


@dataclass(frozen=True)
class RunConfigLoadingEvent(RunLifecycleEvent):
    config_path: str


@dataclass(frozen=True)
class RunConfigLoadedEvent(RunLifecycleEvent):
    pass


@dataclass(frozen=True)
class RunPipelineStartedEvent(RunLifecycleEvent):
    pass


@dataclass(frozen=True)
class RunDeterministicSeedsEvent(RunLifecycleEvent):
    python_seed: int
    numpy_seed: int
    torch_seed: int | None


@dataclass(frozen=True)
class RunBaselineScheduleLoadedEvent(RunLifecycleEvent):
    pass


@dataclass(frozen=True)
class RunAdapterSelectedEvent(RunLifecycleEvent):
    adapter_name: str


@dataclass(frozen=True)
class RunDatasetLoadingEvent(RunLifecycleEvent):
    provider: str


@dataclass(frozen=True)
class RunCalibrationBatchSizesDebugEvent(RunLifecycleEvent):
    preview_count: int
    final_count: int
    total_count: int


@dataclass(frozen=True)
class RunMaskedTokensDebugEvent(RunLifecycleEvent):
    preview_masked: int
    final_masked: int


@dataclass(frozen=True)
class RunPreviewLabelsDebugEvent(RunLifecycleEvent):
    labels: tuple[Any, ...]


@dataclass(frozen=True)
class RunExecutePipelineEvent(RunLifecycleEvent):
    guard_count: int


@dataclass(frozen=True)
class RunLoadModelOnceEvent(RunLifecycleEvent):
    model_id: str


@dataclass(frozen=True)
class RunSnapshotModeEvent(RunLifecycleEvent):
    enabled: bool


@dataclass(frozen=True)
class RunAttemptStartedEvent(RunLifecycleEvent):
    attempt: int
    max_attempts: int | None = None


@dataclass(frozen=True)
class RunRetryAttemptStartedEvent(RunLifecycleEvent):
    attempt: int
    max_attempts: int


@dataclass(frozen=True)
class RunTelemetrySavedEvent(RunLifecycleEvent):
    path: str


@dataclass(frozen=True)
class RunTelemetryFailedEvent(RunLifecycleEvent):
    error: str


@dataclass(frozen=True)
class RunPrimaryMetricSummaryEvent(RunAggregateEvent):
    metric_kind: str
    preview: float
    final: float
    ratio_vs_baseline: float | None = None


@dataclass(frozen=True)
class RunEvaluationReportStartedEvent(RunLifecycleEvent):
    pass


@dataclass(frozen=True)
class RunEvaluationReportPassedEvent(RunLifecycleEvent):
    pass


@dataclass(frozen=True)
class RunEvaluationReportFailedEvent(RunLifecycleEvent):
    gate_codes: tuple[str, ...]


@dataclass(frozen=True)
class RunAutoTuneAdjustmentEvent(RunLifecycleEvent):
    global_k: int
    keep_low: int
    keep_high: int


@dataclass(frozen=True)
class RunRetryExhaustedEvent(RunLifecycleEvent):
    attempt: int


@dataclass(frozen=True)
class RunRetryValidationErrorEvent(RunLifecycleEvent):
    summary: str


@dataclass(frozen=True)
class RunCleanupStatusEvent(RunLifecycleEvent):
    removed: bool


@dataclass(frozen=True)
class RunExecutionFailure:
    """Typed failure contract produced by run orchestration."""

    code: str
    summary: str | None = None
    error: Exception | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunFailureEvent(RunLifecycleEvent):
    failure: RunExecutionFailure


@dataclass(frozen=True)
class RunDeviceResolvedEvent(RunContextEvent):
    requested_device: str
    resolved_device: str


@dataclass(frozen=True)
class RunOutputDirectoryReadyEvent(RunContextEvent):
    run_dir: str
    run_id: str


@dataclass(frozen=True)
class RunEditSelectedEvent(RunContextEvent):
    edit_name: str


@dataclass(frozen=True)
class RunGuardChainResolvedEvent(RunContextEvent):
    guard_names: tuple[str, ...]


@dataclass(frozen=True)
class RunGuardOverheadSummaryEvent(RunAggregateEvent):
    guard_overhead_info: dict[str, Any]
    default_threshold: float


@dataclass(frozen=True)
class RunRetrySummaryEvent(RunAggregateEvent):
    summary: dict[str, Any]


@dataclass(frozen=True)
class RunExecutionResult:
    """Typed result contract produced by run orchestration."""

    report_path: str | None
    timings: dict[str, float]
    timing_summary: TimingSummaryPayload | None = None


@dataclass(frozen=True)
class RunExecutionOutcome:
    """Typed orchestration outcome with event transcript and failure state."""

    ok: bool
    result: RunExecutionResult | None
    failure: RunExecutionFailure | None
    events: tuple[RunExecutionEvent, ...]


RunExecutionObserver = Callable[[RunExecutionEvent], None]


class _RunExecutionHalt(RuntimeError):
    """Internal control-flow sentinel carrying a typed failure."""

    def __init__(self, failure: RunExecutionFailure):
        super().__init__(f"run execution halted: {failure.code}")
        self.failure = failure


@dataclass(frozen=True)
class RunExecutionServices:
    """Typed owner-layer contract for run orchestration helpers."""

    SnapshotRestoreFailed: type[Exception]
    adjust_edit_params: Callable[..., Any]
    assemble_run_report: Callable[..., Any]
    build_snapshot_execution_plan: Callable[..., Any]
    build_provider_dataset_plan: Callable[..., Any]
    execute_guarded_run: Callable[..., Any]
    load_baseline_pairing_evidence: Callable[..., Any]
    materialize_run_dataset: Callable[..., Any]
    free_model_memory: Callable[..., None]
    init_retry_controller: Callable[..., Any]
    load_model_with_cfg: Callable[..., Any]
    persist_run_report_outputs: Callable[..., Any]
    prepare_config_for_run: Callable[..., Any]
    resolve_device_and_output: Callable[..., Any]
    resolve_snapshot_config: Callable[..., Any]
    resolve_snapshot_retry_transition: Callable[..., Any]
    run_bare_control: Callable[..., Any]
    safe_int: Callable[..., int]
    to_serialisable_dict: Callable[..., Any]
    validate_retry_evaluation_report: Callable[..., Any]
    validate_and_harvest_baseline_schedule: Callable[..., Any]
    materialize_baseline_pairing_schedule: Callable[..., Any]
    resolve_tokenizer: Callable[..., Any]
    detect_model_profile: Callable[..., Any]
    get_psutil: Callable[[], Any | None]
    get_torch: Callable[[], Any | None]


def execute_run_request(
    request: RunExecutionRequest,
    *,
    services: RunExecutionServices,
    observer: RunExecutionObserver | None = None,
) -> RunExecutionOutcome:
    from invarlock.core.run_orchestrator_execute import execute_run_request_impl

    return execute_run_request_impl(
        request,
        services=services,
        observer=observer,
    )


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
