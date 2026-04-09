"""Shared helpers for config-driven run orchestration execution."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, NoReturn

from invarlock.core.run_orchestrator_types import (
    RunCleanupStatusEvent,
    RunDiagnosticEvent,
    RunExecutionEvent,
    RunExecutionFailure,
    RunExecutionRequest,
    RunExecutionResult,
    RunExecutionServices,
    RunFailureEvent,
    RunGuardOverheadSummaryEvent,
    RunRetrySummaryEvent,
    TimingSummaryPayload,
    _RunExecutionHalt,
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

_build_run_context_payload_impl: Any | None = None
_build_run_execution_config_payloads_impl: Any | None = None
_resolve_pm_acceptance_range_impl: Any | None = None
_resolve_pm_drift_band_impl: Any | None = None
_resolve_guard_overhead_threshold_impl: Any | None = None
_should_measure_overhead_impl: Any | None = None
np: Any | None = None


def _coerce_float(value: Any, default: float) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return float(default)
    return coerced if math.isfinite(coerced) else float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        return int(default)


RunEventEmitter = Callable[[RunExecutionEvent], None]


@dataclass(frozen=True)
class _RunLossAndSeedState:
    eval_section: Any
    resolved_loss_type: str
    use_mlm: bool
    mask_prob: float
    mask_seed: int
    random_token_prob: float
    original_token_prob: float
    seed_value: int
    seed_bundle: dict[str, int | None]


@dataclass(frozen=True)
class _RunEnvironmentState:
    cfg: Any
    model_profile: Any
    eval_section: Any
    resolved_loss_type: str
    use_mlm: bool
    mask_prob: float
    mask_seed: int
    random_token_prob: float
    original_token_prob: float
    seed_value: int
    seed_bundle: dict[str, int | None]
    profile_label: str | None
    resolved_device: Any
    output_dir: Path
    determinism_meta: dict[str, Any] | None
    run_dir: Path
    run_id: str
    retry_controller: Any | None
    measure_guard_overhead: bool
    skip_overhead: bool
    skip_overhead_source: str | None
    direct_reuse_loaded_model: bool
    emitted_skip_overhead_warning: bool
    tokenizer: Any | None
    tokenizer_hash: str | None
    baseline_report_data: dict[str, Any] | None
    pairing_schedule: dict[str, Any] | None
    requested_preview: int
    requested_final: int
    effective_preview: int
    effective_final: int
    preview_count: int
    final_count: int
    resolved_split: str
    used_fallback_split: bool


@dataclass(frozen=True)
class _RunComponentState:
    adapter: Any
    edit_op: Any
    guards: list[Any]
    run_context: dict[str, Any]
    run_config: Any
    pm_acceptance_range: Any
    pm_drift_band: Any
    guard_overhead_threshold: float


@dataclass(frozen=True)
class _RunDatasetState:
    tokenizer: Any | None
    tokenizer_hash: str | None
    calibration_data: list[dict[str, Any]]
    dataset_meta: dict[str, Any]
    window_plan: dict[str, Any] | None
    preview_count: int
    final_count: int
    effective_preview: int
    effective_final: int
    preview_mask_counts: list[int]
    final_mask_counts: list[int]
    preview_records: list[dict[str, Any]]
    final_records: list[dict[str, Any]]
    resolved_split: str
    used_fallback_split: bool


@dataclass
class _RunExecutionState:
    runner: Any
    auto_config: Any
    edit_config: Any
    model: Any | None
    restore_fn: Any | None
    snapshot_tmpdir: Any | None
    snapshot_provenance: dict[str, bool]
    skip_model_load: bool
    emitted_skip_overhead_warning: bool


@dataclass(frozen=True)
class _AttemptExecutionState:
    attempt: int
    edit_config: Any
    guard_overhead_payload: dict[str, Any] | None
    core_report: Any | None
    should_continue: bool


@dataclass(frozen=True)
class _AttemptDecision:
    report: dict[str, Any]
    timings: dict[str, float]
    report_path_out: str | None
    edit_config: Any
    attempt: int
    should_continue: bool


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
    emit_diagnostic: Callable[..., None],
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


def _cfg_section_value(
    cfg_obj: Any,
    name: str,
    config_value_exceptions: tuple[type[BaseException], ...],
) -> Any:
    section_fn = getattr(cfg_obj, "section", None)
    if callable(section_fn):
        try:
            section = section_fn(name)
        except config_value_exceptions:
            section = None
        if section is not None:
            return section
    try:
        return getattr(cfg_obj, name)
    except config_value_exceptions:
        return None


def _build_outcome_result(
    *,
    capture_timings: bool,
    total_start: float | None,
    timings: dict[str, float],
    report: dict[str, Any],
    report_path_out: str | None,
) -> RunExecutionResult:
    timing_summary: TimingSummaryPayload | None = None
    if capture_timings:
        total_duration = (
            max(0.0, float(perf_counter() - total_start))
            if total_start is not None
            else None
        )
        summary_payload = _build_timing_summary_payload_impl(
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


def _execute_run_pipeline_steps(
    *,
    request: RunExecutionRequest,
    services: RunExecutionServices,
    profile_normalized: str,
    collect_timings: bool,
    total_start: float | None,
    timings: dict[str, float],
    emit: RunEventEmitter,
    emit_diagnostic: Any,
    emit_guard_overhead_summary: Any,
    emit_retry_summary: Any,
    emit_transition: Any,
    halt: Any,
    fail_run: Any,
    record_timed_step: Any,
    cfg_value: Any,
    optional_torch: Any,
    require_torch: Any,
    config_value_exceptions: tuple[type[BaseException], ...],
    numeric_exceptions: tuple[type[BaseException], ...],
    optional_runtime_exceptions: tuple[type[BaseException], ...],
) -> tuple[RunExecutionResult, Any | None]:
    from invarlock.core.api import RunConfig
    from invarlock.core.registry import get_registry
    from invarlock.core.run_orchestrator_execute_attempts import _execute_attempt_loop
    from invarlock.core.run_orchestrator_execute_prepare import (
        _load_dataset_state,
        _prepare_execution_state,
        _prepare_run_environment,
        _resolve_run_components,
    )
    from invarlock.core.runner import CoreRunner

    env_state = _prepare_run_environment(
        config=request.config,
        profile=request.profile,
        profile_normalized=profile_normalized,
        edit=request.edit,
        tier=request.tier,
        probes=request.probes,
        device=request.device,
        out=request.out or "",
        until_pass=bool(request.until_pass),
        max_attempts=int(request.max_attempts),
        timeout=request.timeout,
        baseline=request.baseline,
        determinism_mode=request.determinism_mode,
        determinism_warn_only=bool(request.determinism_warn_only),
        prepare_config_for_run_fn=services.prepare_config_for_run,
        detect_model_profile_fn=services.detect_model_profile,
        resolve_device_and_output_fn=services.resolve_device_and_output,
        init_retry_controller_fn=services.init_retry_controller,
        load_baseline_pairing_evidence_fn=services.load_baseline_pairing_evidence,
        safe_int_fn=services.safe_int,
        optional_torch=optional_torch,
        require_torch=require_torch,
        cfg_value=cfg_value,
        emit=emit,
        emit_diagnostic=emit_diagnostic,
        config_value_exceptions=config_value_exceptions,
        numeric_exceptions=numeric_exceptions,
        optional_runtime_exceptions=optional_runtime_exceptions,
    )
    component_state = _resolve_run_components(
        cfg=env_state.cfg,
        profile=request.profile,
        eval_device_override=request.eval_device_override,
        pairing_schedule=env_state.pairing_schedule,
        seed_bundle=env_state.seed_bundle,
        run_id=env_state.run_id,
        baseline_report_data=env_state.baseline_report_data,
        model_profile=env_state.model_profile,
        resolved_loss_type=env_state.resolved_loss_type,
        tiny_relax_enabled=bool(request.tiny_relax_enabled),
        resolved_device=env_state.resolved_device,
        eval_section=env_state.eval_section,
        run_dir=env_state.run_dir,
        get_registry_fn=get_registry,
        run_config_type=RunConfig,
        to_serialisable_dict_fn=services.to_serialisable_dict,
        cfg_value=cfg_value,
        emit=emit,
        emit_diagnostic=emit_diagnostic,
        halt=halt,
    )
    dataset_state = _load_dataset_state(
        cfg=env_state.cfg,
        model_profile=env_state.model_profile,
        resolved_device=env_state.resolved_device,
        profile=request.profile,
        profile_normalized=profile_normalized,
        requested_preview=env_state.requested_preview,
        requested_final=env_state.requested_final,
        effective_preview=env_state.effective_preview,
        effective_final=env_state.effective_final,
        use_mlm=env_state.use_mlm,
        mask_prob=env_state.mask_prob,
        mask_seed=env_state.mask_seed,
        random_token_prob=env_state.random_token_prob,
        original_token_prob=env_state.original_token_prob,
        resolved_loss_type=env_state.resolved_loss_type,
        tier=request.tier,
        baseline_report_data=env_state.baseline_report_data,
        tokenizer=env_state.tokenizer,
        tokenizer_hash=env_state.tokenizer_hash,
        resolved_split=env_state.resolved_split,
        pairing_schedule=env_state.pairing_schedule,
        collect_timings=collect_timings,
        timings=timings,
        run_context=component_state.run_context,
        materialize_run_dataset_fn=services.materialize_run_dataset,
        validate_and_harvest_baseline_schedule_fn=(
            services.validate_and_harvest_baseline_schedule
        ),
        materialize_baseline_pairing_schedule_fn=(
            services.materialize_baseline_pairing_schedule
        ),
        resolve_tokenizer_fn=services.resolve_tokenizer,
        build_provider_dataset_plan_fn=services.build_provider_dataset_plan,
        emit=emit,
        emit_transition=emit_transition,
        fail_run=fail_run,
    )
    execution_state = _prepare_execution_state(
        cfg=env_state.cfg,
        model_profile=env_state.model_profile,
        profile_normalized=profile_normalized,
        resolved_device=env_state.resolved_device,
        run_dir=env_state.run_dir,
        run_id=env_state.run_id,
        adapter=component_state.adapter,
        edit_op=component_state.edit_op,
        guards=component_state.guards,
        prefer_local_files_only=request.prefer_local_files_only,
        skip_overhead=env_state.skip_overhead,
        skip_overhead_source=env_state.skip_overhead_source,
        direct_reuse_loaded_model=env_state.direct_reuse_loaded_model,
        emitted_skip_overhead_warning=env_state.emitted_skip_overhead_warning,
        retry_controller=env_state.retry_controller,
        cfg_value=cfg_value,
        emit=emit,
        emit_transition=emit_transition,
        record_timed_step=record_timed_step,
        load_model_with_cfg_fn=services.load_model_with_cfg,
        build_snapshot_execution_plan_fn=services.build_snapshot_execution_plan,
        resolve_snapshot_config_fn=services.resolve_snapshot_config,
        resolve_snapshot_retry_transition_fn=services.resolve_snapshot_retry_transition,
        free_model_memory_fn=services.free_model_memory,
        core_runner_type=CoreRunner,
        optional_runtime_exceptions=optional_runtime_exceptions,
    )
    attempt_decision = _execute_attempt_loop(
        execution_state=execution_state,
        cfg=env_state.cfg,
        adapter=component_state.adapter,
        edit_op=component_state.edit_op,
        guards=component_state.guards,
        run_config=component_state.run_config,
        calibration_data=dataset_state.calibration_data,
        preview_count=dataset_state.preview_count,
        final_count=dataset_state.final_count,
        resolved_device=env_state.resolved_device,
        profile_normalized=profile_normalized,
        guard_overhead_threshold=component_state.guard_overhead_threshold,
        skip_overhead=env_state.skip_overhead,
        skip_overhead_source=env_state.skip_overhead_source,
        measure_guard_overhead=env_state.measure_guard_overhead,
        resolved_loss_type=env_state.resolved_loss_type,
        prefer_local_files_only=request.prefer_local_files_only,
        retry_controller=env_state.retry_controller,
        max_attempts=int(request.max_attempts),
        seed_bundle=env_state.seed_bundle,
        seed_value=env_state.seed_value,
        build_restore_failure_attempt_summary_fn=(
            _build_restore_failure_attempt_summary_impl
        ),
        decide_failed_retry_transition_fn=_decide_failed_retry_transition_impl,
        free_model_memory_fn=services.free_model_memory,
        adjust_edit_params_fn=services.adjust_edit_params,
        run_bare_control_fn=services.run_bare_control,
        execute_guarded_run_fn=services.execute_guarded_run,
        snapshot_restore_failed_type=services.SnapshotRestoreFailed,
        assemble_run_report_fn=services.assemble_run_report,
        persist_run_report_outputs_fn=services.persist_run_report_outputs,
        validate_retry_evaluation_report_fn=services.validate_retry_evaluation_report,
        resolve_retry_validation_transition_fn=(
            _resolve_retry_validation_transition_impl
        ),
        record_retry_attempt_fn=_record_retry_attempt_impl,
        emit=emit,
        emit_diagnostic=emit_diagnostic,
        emit_guard_overhead_summary=emit_guard_overhead_summary,
        emit_transition=emit_transition,
        halt=halt,
        fail_run=fail_run,
        record_timed_step=record_timed_step,
        timings=timings,
        profile=request.profile,
        baseline=request.baseline,
        edit_label=request.edit_label,
        metric_kind=request.metric_kind,
        export_model_requested=bool(request.export_model_requested),
        export_dir_override=request.export_dir,
        telemetry=bool(request.telemetry),
        tokenizer=dataset_state.tokenizer,
        tokenizer_hash=dataset_state.tokenizer_hash,
        resolved_split=dataset_state.resolved_split,
        effective_preview=dataset_state.effective_preview,
        effective_final=dataset_state.effective_final,
        preview_records=dataset_state.preview_records,
        final_records=dataset_state.final_records,
        preview_mask_counts=dataset_state.preview_mask_counts,
        final_mask_counts=dataset_state.final_mask_counts,
        use_mlm=env_state.use_mlm,
        used_fallback_split=dataset_state.used_fallback_split,
        baseline_report_data=env_state.baseline_report_data,
        window_plan=dataset_state.window_plan,
        model_profile=env_state.model_profile,
        determinism_meta=env_state.determinism_meta,
        pm_acceptance_range=component_state.pm_acceptance_range,
        pm_drift_band=component_state.pm_drift_band,
        run_dir=env_state.run_dir,
        cfg_value=cfg_value,
        optional_runtime_exceptions=optional_runtime_exceptions,
    )
    emit_retry_summary(env_state.retry_controller)
    return (
        _build_outcome_result(
            capture_timings=bool(request.capture_timings),
            total_start=total_start,
            timings=attempt_decision.timings,
            report=attempt_decision.report,
            report_path_out=attempt_decision.report_path_out,
        ),
        execution_state.snapshot_tmpdir,
    )
