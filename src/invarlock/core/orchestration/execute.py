"""Internal execution implementation for config-driven run orchestration."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import partial
from time import perf_counter
from typing import Any, NoReturn

import numpy as np  # noqa: F401

import invarlock.core.orchestration.helpers as _execute_helpers
from invarlock.core.exceptions import InvarlockError
from invarlock.core.orchestration.helpers import (
    RunEventEmitter as RunEventEmitter,
)
from invarlock.core.orchestration.helpers import (
    _cfg_section_value,
)
from invarlock.core.orchestration.helpers import (
    _coerce_float as _coerce_float,
)
from invarlock.core.orchestration.helpers import (
    _coerce_int as _coerce_int,
)
from invarlock.core.retry import (
    build_restore_failure_attempt_summary,
    decide_failed_retry_transition,
    record_retry_attempt,
)
from invarlock.core.retry import (
    resolve_retry_validation_transition as _resolve_retry_validation_transition_impl,
)
from invarlock.core.run_orchestrator import (
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
    RunGuardMetricImpactSummaryEvent,
    RunRetrySummaryEvent,
    _RunExecutionHalt,
)
from invarlock.core.run_policy import (
    build_run_context_payload as _build_run_context_payload_impl,
)
from invarlock.core.run_policy import (
    build_run_execution_config_payloads as _build_run_execution_config_payloads_impl,
)
from invarlock.core.run_policy import (
    build_timing_summary_payload as _build_timing_summary_payload_impl,
)
from invarlock.core.run_policy import (
    resolve_guard_metric_degradation_limit as _resolve_guard_metric_degradation_limit_impl,
)
from invarlock.core.run_policy import (
    resolve_pm_acceptance_range as _resolve_pm_acceptance_range_impl,
)
from invarlock.core.run_policy import (
    resolve_pm_drift_band as _resolve_pm_drift_band_impl,
)
from invarlock.core.run_policy import (
    should_measure_metric_impact as _should_measure_metric_impact_impl,
)

# class RunExecutionEvent
# class RunLifecycleEvent
# class RunDiagnosticEvent
# class RunContextEvent
# class RunAggregateEvent
# class RunFailureEvent
# class RunPrimaryMetricSummaryEvent
# Typed contracts live in `invarlock.core.run_orchestrator` and are
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


def _emit_run_guard_metric_impact_summary(
    emit: RunEventEmitter,
    guard_metric_impact_info: dict[str, Any],
    *,
    default_limit: float,
) -> None:
    emit(
        RunGuardMetricImpactSummaryEvent(
            guard_metric_impact_info=guard_metric_impact_info,
            default_limit=default_limit,
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


def _build_outcome_result(
    *,
    capture_timings: bool,
    total_start: float | None,
    timings: dict[str, float],
    report: dict[str, Any],
    report_path_out: str | None,
    build_timing_summary_payload_fn: Any,
) -> RunExecutionResult:
    timing_summary: Any | None = None
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


def _execute_run_pipeline_steps(
    *,
    request: RunExecutionRequest,
    services: RunExecutionServices,
    profile_normalized: str,
    collect_timings: bool,
    total_start: float | None,
    timings: dict[str, float],
    emit: Any,
    emit_diagnostic: Any,
    emit_guard_metric_impact_summary: Any,
    emit_retry_summary: Any,
    emit_transition: Any,
    halt: Any,
    fail_run: Any,
    record_timed_step: Any,
    cfg_value: Any,
    optional_torch: Any,
    require_torch: Any,
    build_timing_summary_payload_fn: Any,
    resolve_retry_validation_transition_fn: Any,
    config_value_exceptions: tuple[type[BaseException], ...],
    numeric_exceptions: tuple[type[BaseException], ...],
    optional_runtime_exceptions: tuple[type[BaseException], ...],
) -> tuple[RunExecutionResult, Any | None]:
    from invarlock.core.api import RunConfig
    from invarlock.core.orchestration.attempts import _execute_attempt_loop
    from invarlock.core.orchestration.environment import (
        _prepare_run_environment,
    )
    from invarlock.core.orchestration.execution import (
        _load_dataset_state,
        _prepare_execution_state,
        _resolve_run_components,
    )
    from invarlock.core.registry import get_registry
    from invarlock.core.runner import CoreRunner

    env_state = _prepare_run_environment(
        config=request.config,
        profile=request.profile,
        profile_normalized=profile_normalized,
        edit=request.edit,
        tier=request.tier,
        probes=request.probes,
        resolved_config_out=request.resolved_config_out,
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
        skip_guard_metric_impact=env_state.skip_guard_metric_impact,
        skip_guard_metric_impact_source=env_state.skip_guard_metric_impact_source,
        direct_reuse_loaded_model=env_state.direct_reuse_loaded_model,
        emitted_skip_guard_metric_impact_warning=env_state.emitted_skip_guard_metric_impact_warning,
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
        guard_metric_degradation_limit=component_state.guard_metric_degradation_limit,
        skip_guard_metric_impact=env_state.skip_guard_metric_impact,
        skip_guard_metric_impact_source=env_state.skip_guard_metric_impact_source,
        measure_guard_metric_impact=env_state.measure_guard_metric_impact,
        resolved_loss_type=env_state.resolved_loss_type,
        prefer_local_files_only=request.prefer_local_files_only,
        retry_controller=env_state.retry_controller,
        max_attempts=int(request.max_attempts),
        seed_bundle=env_state.seed_bundle,
        seed_value=env_state.seed_value,
        build_restore_failure_attempt_summary_fn=build_restore_failure_attempt_summary,
        decide_failed_retry_transition_fn=decide_failed_retry_transition,
        free_model_memory_fn=services.free_model_memory,
        adjust_edit_params_fn=services.adjust_edit_params,
        run_bare_control_fn=services.run_bare_control,
        execute_guarded_run_fn=services.execute_guarded_run,
        snapshot_restore_failed_type=services.SnapshotRestoreFailed,
        assemble_run_report_fn=services.assemble_run_report,
        persist_run_report_outputs_fn=services.persist_run_report_outputs,
        validate_retry_evaluation_report_fn=services.validate_retry_evaluation_report,
        resolve_retry_validation_transition_fn=resolve_retry_validation_transition_fn,
        record_retry_attempt_fn=record_retry_attempt,
        emit=emit,
        emit_diagnostic=emit_diagnostic,
        emit_guard_metric_impact_summary=emit_guard_metric_impact_summary,
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
            build_timing_summary_payload_fn=build_timing_summary_payload_fn,
        ),
        execution_state.snapshot_tmpdir,
    )


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
    _emit_guard_metric_impact_summary = partial(
        _emit_run_guard_metric_impact_summary, _emit
    )
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
        _execute_helpers._resolve_guard_metric_degradation_limit_impl = (
            _resolve_guard_metric_degradation_limit_impl
        )
        _execute_helpers._should_measure_metric_impact_impl = (
            _should_measure_metric_impact_impl
        )
        outcome_result, snapshot_tmpdir = _execute_run_pipeline_steps(
            request=request,
            services=services,
            profile_normalized=profile_normalized,
            collect_timings=collect_timings,
            total_start=total_start,
            timings=timings,
            emit=_emit,
            emit_diagnostic=_emit_diagnostic,
            emit_guard_metric_impact_summary=_emit_guard_metric_impact_summary,
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
