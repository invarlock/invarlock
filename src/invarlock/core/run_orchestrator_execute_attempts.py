"""Attempt execution helpers for config-driven run orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock.core import run_orchestrator_execute_attempt_results as _attempt_results
from invarlock.core.run_orchestrator import (
    RunAttemptStartedEvent,
    RunRetryAttemptStartedEvent,
)
from invarlock.core.run_orchestrator_execute_helpers import (
    RunEventEmitter,
    _AttemptDecision,
    _AttemptExecutionState,
    _RunExecutionState,
)

_emit_primary_metric_summary_from_report = (
    _attempt_results._emit_primary_metric_summary_from_report
)
_enforce_guard_overhead_budget = _attempt_results._enforce_guard_overhead_budget
_handle_retry_validation = _attempt_results._handle_retry_validation
_maybe_export_model_artifacts = _attempt_results._maybe_export_model_artifacts
_process_attempt_result = _attempt_results._process_attempt_result
_resolve_export_model_dir = _attempt_results._resolve_export_model_dir
_should_export_model = _attempt_results._should_export_model


def _emit_attempt_start(
    *,
    emit: RunEventEmitter,
    retry_controller: Any | None,
    attempt: int,
    max_attempts: int,
) -> None:
    if retry_controller:
        emit(
            RunAttemptStartedEvent(
                attempt=int(attempt),
                max_attempts=int(max_attempts),
            )
        )
        if attempt > 1:
            emit(
                RunRetryAttemptStartedEvent(
                    attempt=int(attempt),
                    max_attempts=int(max_attempts),
                )
            )
        return
    if attempt > 1:
        emit(RunAttemptStartedEvent(attempt=int(attempt)))


def _build_skipped_guard_overhead_payload(
    *,
    guard_overhead_threshold: float,
    skip_overhead_source: str | None,
) -> dict[str, Any]:
    skip_reason = (
        "context.run.skip_overhead_check"
        if skip_overhead_source == "config:context.run.skip_overhead_check"
        else "context.eval.skip_overhead_check"
    )
    return {
        "overhead_threshold": guard_overhead_threshold,
        "evaluated": False,
        "passed": True,
        "skipped": True,
        "skip_reason": skip_reason,
        "mode": "skipped",
        "source": skip_overhead_source or "config:context.run.skip_overhead_check",
        "diagnostics": [
            {
                "kind": "guard_overhead_info",
                "severity": "info",
                "message": "Overhead check skipped via config policy",
                "details": {},
            }
        ],
        "checks": {},
    }


def _execute_attempt_core(
    *,
    attempt: int,
    max_attempts: int,
    retry_controller: Any | None,
    seed_bundle: dict[str, int | None],
    seed_value: int,
    edit_op: Any,
    cfg: Any,
    adapter: Any,
    run_config: Any,
    guards: list[Any],
    calibration_data: list[dict[str, Any]],
    preview_count: int,
    final_count: int,
    resolved_device: Any,
    profile_normalized: str,
    guard_overhead_threshold: float,
    skip_overhead: bool,
    skip_overhead_source: str | None,
    measure_guard_overhead: bool,
    resolved_loss_type: str,
    prefer_local_files_only: bool,
    execution_state: _RunExecutionState,
    adjust_edit_params_fn: Any,
    run_bare_control_fn: Any,
    execute_guarded_run_fn: Any,
    snapshot_restore_failed_type: type[BaseException],
    build_restore_failure_attempt_summary_fn: Any,
    decide_failed_retry_transition_fn: Any,
    free_model_memory_fn: Any,
    emit: RunEventEmitter,
    emit_transition: Any,
    emit_diagnostic: Any,
    halt: Any,
    record_timed_step: Any,
) -> _AttemptExecutionState:
    from invarlock.core.determinism_policy import set_seed

    set_seed(int(seed_bundle.get("python") or seed_value))
    _emit_attempt_start(
        emit=emit,
        retry_controller=retry_controller,
        attempt=attempt,
        max_attempts=max_attempts,
    )
    edit_config = execution_state.edit_config
    if retry_controller and attempt > 1:
        adjustment = adjust_edit_params_fn(edit_op.name, edit_config, attempt, None)
        edit_config = adjustment.params
        for diagnostic in adjustment.diagnostics:
            emit_transition("retry_adjustment", diagnostic)
    guard_overhead_payload: dict[str, Any] | None = None
    try:
        if skip_overhead and profile_normalized in {"ci", "release"}:
            guard_overhead_payload = _build_skipped_guard_overhead_payload(
                guard_overhead_threshold=guard_overhead_threshold,
                skip_overhead_source=skip_overhead_source,
            )
        elif measure_guard_overhead:
            guard_overhead_payload = run_bare_control_fn(
                adapter=adapter,
                edit_op=edit_op,
                cfg=cfg,
                model=execution_state.model,
                run_config=run_config,
                calibration_data=calibration_data,
                auto_config=execution_state.auto_config,
                edit_config=edit_config,
                preview_count=preview_count,
                final_count=final_count,
                seed_bundle=seed_bundle,
                resolved_device=resolved_device,
                restore_fn=execution_state.restore_fn,
                resolved_loss_type=resolved_loss_type,
                overhead_threshold=guard_overhead_threshold,
                profile_normalized=profile_normalized,
                snapshot_provenance=execution_state.snapshot_provenance,
                skip_model_load=execution_state.skip_model_load,
                prefer_local_files_only=prefer_local_files_only,
            )
        with record_timed_step("execute"):
            core_report, execution_state.model = execute_guarded_run_fn(
                runner=execution_state.runner,
                adapter=adapter,
                model=execution_state.model,
                cfg=cfg,
                edit_op=edit_op,
                run_config=run_config,
                guards=guards,
                calibration_data=calibration_data,
                auto_config=execution_state.auto_config,
                edit_config=edit_config,
                preview_count=preview_count,
                final_count=final_count,
                restore_fn=execution_state.restore_fn,
                resolved_device=resolved_device,
                profile_normalized=profile_normalized,
                snapshot_provenance=execution_state.snapshot_provenance,
                skip_model_load=execution_state.skip_model_load,
                prefer_local_files_only=prefer_local_files_only,
            )
    except snapshot_restore_failed_type as exc:
        reload_fallback_already_attempted = bool(
            execution_state.snapshot_provenance.get("restore_failed")
        )
        execution_state.snapshot_provenance["restore_failed"] = True
        free_model_memory_fn(execution_state.model)
        execution_state.model = None
        execution_state.restore_fn = None
        emit_diagnostic(code="snapshot_restore_fallback", error=str(exc))
        if retry_controller is None and not reload_fallback_already_attempted:
            return _AttemptExecutionState(
                attempt=attempt,
                edit_config=edit_config,
                guard_overhead_payload=None,
                core_report=None,
                model=execution_state.model,
                should_continue=True,
            )
        retry_transition = decide_failed_retry_transition_fn(
            retry_controller,
            attempt=attempt,
            attempt_summary=build_restore_failure_attempt_summary_fn(),
            edit_config=edit_config,
            passed=False,
        )
        for diagnostic in retry_transition.diagnostics:
            emit_transition("retry_failure", diagnostic)
        if retry_transition.should_retry:
            return _AttemptExecutionState(
                attempt=retry_transition.next_attempt,
                edit_config=edit_config,
                guard_overhead_payload=None,
                core_report=None,
                model=execution_state.model,
                should_continue=True,
            )
        halt("snapshot_restore_failed", error=exc)
    core_status = str(getattr(core_report, "status", "") or "").strip().lower()
    if core_status in {"failed", "error"}:
        core_error = str(getattr(core_report, "error", "") or "").strip()
        if not core_error:
            core_error = (
                f"Guarded run failed before report assembly (status: {core_status})."
            )
        halt(
            "pipeline_failed",
            summary=core_error,
            status=core_status,
            phase="guarded_run",
        )
    return _AttemptExecutionState(
        attempt=attempt,
        edit_config=edit_config,
        guard_overhead_payload=guard_overhead_payload,
        core_report=core_report,
        model=execution_state.model,
        should_continue=False,
    )


def _execute_attempt_loop(
    *,
    execution_state: _RunExecutionState,
    cfg: Any,
    adapter: Any,
    edit_op: Any,
    guards: list[Any],
    run_config: Any,
    calibration_data: list[dict[str, Any]],
    preview_count: int,
    final_count: int,
    resolved_device: Any,
    profile_normalized: str,
    guard_overhead_threshold: float,
    skip_overhead: bool,
    skip_overhead_source: str | None,
    measure_guard_overhead: bool,
    resolved_loss_type: str,
    prefer_local_files_only: bool,
    retry_controller: Any | None,
    max_attempts: int,
    seed_bundle: dict[str, int | None],
    seed_value: int,
    build_restore_failure_attempt_summary_fn: Any,
    decide_failed_retry_transition_fn: Any,
    free_model_memory_fn: Any,
    adjust_edit_params_fn: Any,
    run_bare_control_fn: Any,
    execute_guarded_run_fn: Any,
    snapshot_restore_failed_type: type[BaseException],
    assemble_run_report_fn: Any,
    persist_run_report_outputs_fn: Any,
    validate_retry_evaluation_report_fn: Any,
    resolve_retry_validation_transition_fn: Any,
    record_retry_attempt_fn: Any,
    emit: RunEventEmitter,
    emit_diagnostic: Any,
    emit_guard_overhead_summary: Any,
    emit_transition: Any,
    halt: Any,
    fail_run: Any,
    record_timed_step: Any,
    timings: dict[str, float],
    profile: str | None,
    baseline: str | None,
    edit_label: str | None,
    metric_kind: str | None,
    export_model_requested: bool,
    export_dir_override: str | None,
    telemetry: bool,
    tokenizer: Any | None,
    tokenizer_hash: str | None,
    resolved_split: str,
    effective_preview: int,
    effective_final: int,
    preview_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    preview_mask_counts: list[int],
    final_mask_counts: list[int],
    use_mlm: bool,
    used_fallback_split: bool,
    baseline_report_data: dict[str, Any] | None,
    window_plan: dict[str, Any] | None,
    model_profile: Any,
    determinism_meta: dict[str, Any] | None,
    pm_acceptance_range: Any,
    pm_drift_band: Any,
    run_dir: Path,
    cfg_value: Any,
    optional_runtime_exceptions: tuple[type[BaseException], ...],
) -> _AttemptDecision:
    attempt = 1
    report_path_out: str | None = None
    while True:
        attempt_state = _execute_attempt_core(
            attempt=attempt,
            max_attempts=max_attempts,
            retry_controller=retry_controller,
            seed_bundle=seed_bundle,
            seed_value=seed_value,
            edit_op=edit_op,
            cfg=cfg,
            adapter=adapter,
            run_config=run_config,
            guards=guards,
            calibration_data=calibration_data,
            preview_count=preview_count,
            final_count=final_count,
            resolved_device=resolved_device,
            profile_normalized=profile_normalized,
            guard_overhead_threshold=guard_overhead_threshold,
            skip_overhead=skip_overhead,
            skip_overhead_source=skip_overhead_source,
            measure_guard_overhead=measure_guard_overhead,
            resolved_loss_type=resolved_loss_type,
            prefer_local_files_only=prefer_local_files_only,
            execution_state=execution_state,
            adjust_edit_params_fn=adjust_edit_params_fn,
            run_bare_control_fn=run_bare_control_fn,
            execute_guarded_run_fn=execute_guarded_run_fn,
            snapshot_restore_failed_type=snapshot_restore_failed_type,
            build_restore_failure_attempt_summary_fn=build_restore_failure_attempt_summary_fn,
            decide_failed_retry_transition_fn=decide_failed_retry_transition_fn,
            free_model_memory_fn=free_model_memory_fn,
            emit=emit,
            emit_transition=emit_transition,
            emit_diagnostic=emit_diagnostic,
            halt=halt,
            record_timed_step=record_timed_step,
        )
        execution_state.edit_config = attempt_state.edit_config
        if attempt_state.should_continue:
            attempt = attempt_state.attempt
            continue
        decision = _process_attempt_result(
            attempt_state=attempt_state,
            timings=timings,
            report_path_out=report_path_out,
            cfg=cfg,
            profile_normalized=profile_normalized,
            profile=profile,
            baseline=baseline,
            edit_label=edit_label,
            metric_kind=metric_kind,
            export_model_requested=export_model_requested,
            export_dir_override=export_dir_override,
            telemetry=telemetry,
            resolved_loss_type=resolved_loss_type,
            tokenizer=tokenizer,
            tokenizer_hash=tokenizer_hash,
            resolved_split=resolved_split,
            preview_count=preview_count,
            final_count=final_count,
            effective_preview=effective_preview,
            effective_final=effective_final,
            preview_records=preview_records,
            final_records=final_records,
            preview_mask_counts=preview_mask_counts,
            final_mask_counts=final_mask_counts,
            use_mlm=use_mlm,
            used_fallback_split=used_fallback_split,
            baseline_report_data=baseline_report_data,
            window_plan=window_plan,
            model_profile=model_profile,
            determinism_meta=determinism_meta,
            guard_overhead_threshold=guard_overhead_threshold,
            pm_acceptance_range=pm_acceptance_range,
            pm_drift_band=pm_drift_band,
            seed_bundle=seed_bundle,
            run_dir=run_dir,
            run_config=run_config,
            auto_config=execution_state.auto_config,
            resolved_device=resolved_device,
            snapshot_provenance=execution_state.snapshot_provenance,
            edit_op=edit_op,
            adapter=adapter,
            model=attempt_state.model,
            measure_guard_overhead=measure_guard_overhead,
            retry_controller=retry_controller,
            validate_retry_evaluation_report_fn=validate_retry_evaluation_report_fn,
            resolve_retry_validation_transition_fn=resolve_retry_validation_transition_fn,
            record_retry_attempt_fn=record_retry_attempt_fn,
            persist_run_report_outputs_fn=persist_run_report_outputs_fn,
            assemble_run_report_fn=assemble_run_report_fn,
            cfg_value=cfg_value,
            emit=emit,
            emit_diagnostic=emit_diagnostic,
            emit_guard_overhead_summary=emit_guard_overhead_summary,
            emit_transition=emit_transition,
            halt=halt,
            fail_run=fail_run,
            optional_runtime_exceptions=optional_runtime_exceptions,
        )
        execution_state.edit_config = decision.edit_config
        if decision.should_continue:
            report_path_out = decision.report_path_out
            timings = dict(decision.timings)
            attempt = decision.attempt
            continue
        return decision


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
