"""Attempt execution helpers for config-driven run orchestration."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from invarlock.core.exceptions import InvarlockError
from invarlock.core.run_orchestrator import (
    RunAttemptStartedEvent,
    RunAutoTuneAdjustmentEvent,
    RunEvaluationReportFailedEvent,
    RunEvaluationReportPassedEvent,
    RunEvaluationReportStartedEvent,
    RunPrimaryMetricSummaryEvent,
    RunRetryAttemptStartedEvent,
    RunRetryExhaustedEvent,
    RunRetryValidationErrorEvent,
    RunTelemetryFailedEvent,
    RunTelemetrySavedEvent,
)
from invarlock.core.run_orchestrator_execute_helpers import (
    RunEventEmitter,
    _AttemptDecision,
    _AttemptExecutionState,
    _RunExecutionState,
)


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


def _emit_primary_metric_summary_from_report(
    *,
    report: dict[str, Any],
    emit: RunEventEmitter,
) -> None:
    try:
        pm_obj = report.get("metrics", {}).get("primary_metric")
    except (AttributeError, TypeError, KeyError):
        pm_obj = None
    if not isinstance(pm_obj, dict) or not pm_obj:
        return
    try:
        pm_kind = str(pm_obj.get("kind", "primary")).lower()
        pm_prev = pm_obj.get("preview")
        pm_fin = pm_obj.get("final")
        ratio_vs_base = pm_obj.get("ratio_vs_baseline")
        if isinstance(pm_prev, int | float) and isinstance(pm_fin, int | float):
            emit(
                RunPrimaryMetricSummaryEvent(
                    metric_kind=pm_kind,
                    preview=float(pm_prev),
                    final=float(pm_fin),
                    ratio_vs_baseline=(
                        float(ratio_vs_base)
                        if isinstance(ratio_vs_base, int | float)
                        and math.isfinite(ratio_vs_base)
                        else None
                    ),
                )
            )
    except (TypeError, ValueError):
        return


def _should_export_model(
    *,
    output_cfg: Any,
    export_model_requested: bool,
) -> bool:
    save_model_cfg = False
    try:
        if isinstance(output_cfg, dict):
            save_model_cfg = bool(output_cfg.get("save_model", False))
        else:
            save_model_cfg = bool(getattr(output_cfg, "save_model", False))
    except (AttributeError, TypeError):
        save_model_cfg = False
    return bool(export_model_requested) or save_model_cfg


def _resolve_export_model_dir(
    *,
    output_cfg: Any,
    run_dir: Path,
    export_dir_override: str | None,
    optional_runtime_exceptions: tuple[type[BaseException], ...],
) -> Path:
    export_dir: Path | None = None
    try:
        model_dir_cfg = None
        if isinstance(output_cfg, dict):
            model_dir_cfg = output_cfg.get("model_dir") or output_cfg.get("model_path")
        elif output_cfg is not None:
            model_dir_cfg = getattr(output_cfg, "model_dir", None) or getattr(
                output_cfg,
                "model_path",
                None,
            )
        if model_dir_cfg:
            candidate = Path(str(model_dir_cfg))
            export_dir = candidate if candidate.is_absolute() else (run_dir / candidate)
    except optional_runtime_exceptions:
        export_dir = None
    if export_dir is None and isinstance(export_dir_override, str):
        if export_dir_override.strip():
            candidate = Path(export_dir_override.strip())
            export_dir = candidate if candidate.is_absolute() else (run_dir / candidate)
    if export_dir is not None:
        return export_dir
    try:
        if isinstance(output_cfg, dict):
            resolved_export_subdir = str(output_cfg.get("model_subdir", "model"))
        else:
            resolved_export_subdir = str(getattr(output_cfg, "model_subdir", "model"))
    except optional_runtime_exceptions:
        resolved_export_subdir = "model"
    return run_dir / resolved_export_subdir


def _maybe_export_model_artifacts(
    *,
    cfg: Any,
    run_dir: Path,
    report: dict[str, Any],
    adapter: Any,
    model: Any | None,
    tokenizer: Any | None,
    export_model_requested: bool,
    export_dir_override: str | None,
    cfg_value: Any,
    emit_diagnostic: Any,
    optional_runtime_exceptions: tuple[type[BaseException], ...],
) -> None:
    output_cfg = cfg_value(cfg, "output") or {}
    if not _should_export_model(
        output_cfg=output_cfg,
        export_model_requested=export_model_requested,
    ):
        return
    try:
        export_dir = _resolve_export_model_dir(
            output_cfg=output_cfg,
            run_dir=run_dir,
            export_dir_override=export_dir_override,
            optional_runtime_exceptions=optional_runtime_exceptions,
        )
        ok = False
        if hasattr(adapter, "save_pretrained") and model is not None:
            ok = bool(adapter.save_pretrained(model, export_dir))
        if not ok:
            emit_diagnostic(code="export_adapter_directory_missing")
            return
        save_tokenizer = getattr(tokenizer, "save_pretrained", None)
        if callable(save_tokenizer):
            try:
                save_tokenizer(str(export_dir))
            except optional_runtime_exceptions:
                emit_diagnostic(code="export_tokenizer_missing")
        report["artifacts"]["checkpoint_path"] = str(export_dir)
    except optional_runtime_exceptions:
        emit_diagnostic(code="export_failed")


def _enforce_guard_overhead_budget(
    *,
    report: dict[str, Any],
    run_config: Any,
    measure_guard_overhead: bool,
    guard_overhead_threshold: float,
    emit_guard_overhead_summary: Any,
    halt: Any,
) -> None:
    guard_overhead_info = report.get("guard_overhead")
    if not guard_overhead_info:
        return
    emit_guard_overhead_summary(
        guard_overhead_info,
        default_threshold=guard_overhead_threshold,
    )
    threshold_fraction = float(
        guard_overhead_info.get("overhead_threshold", guard_overhead_threshold)
        or guard_overhead_threshold
    )
    if guard_overhead_info.get("passed", True):
        return
    loss_type_ctx = None
    try:
        loss_type_ctx = (
            run_config.context.get("eval", {}).get("loss", {}).get("resolved_type")
        )
    except (AttributeError, KeyError, TypeError):
        loss_type_ctx = None
    if (
        measure_guard_overhead
        and guard_overhead_info.get("evaluated", False)
        and str(loss_type_ctx).lower() != "mlm"
    ):
        halt(
            "guard_overhead_budget_exceeded",
            threshold_fraction=float(threshold_fraction),
        )


def _handle_retry_validation(
    *,
    retry_controller: Any | None,
    baseline: str | None,
    report: dict[str, Any],
    baseline_report_data: dict[str, Any] | None,
    edit_config: Any,
    attempt: int,
    validate_retry_evaluation_report_fn: Any,
    resolve_retry_validation_transition_fn: Any,
    record_retry_attempt_fn: Any,
    emit: RunEventEmitter,
    emit_transition: Any,
    emit_diagnostic: Any,
) -> tuple[Any, int, bool]:
    if retry_controller and baseline:
        emit(RunEvaluationReportStartedEvent())
        retry_validation = validate_retry_evaluation_report_fn(
            report=report,
            baseline_report_data=baseline_report_data,
            baseline_path=Path(baseline),
        )
        if retry_validation.telemetry_summary:
            emit_diagnostic(
                code="retry_validation_telemetry_summary",
                summary=retry_validation.telemetry_summary,
            )
        retry_decision = resolve_retry_validation_transition_fn(
            retry_controller,
            attempt=attempt,
            validation_result=retry_validation,
            edit_config=edit_config,
        )
        retry_disposition = str(getattr(retry_decision, "status", "error") or "error")
        retry_gate_codes = tuple(
            str(item)
            for item in (getattr(retry_decision, "validation_gates", ()) or ())
        )
        retry_error = getattr(retry_decision, "error", None)
        retry_summary = str(
            getattr(retry_error, "message", None) or "Retry validation failed"
        )
        if retry_disposition == "passed":
            emit(RunEvaluationReportPassedEvent())
            return edit_config, attempt, False
        if retry_disposition in {"retry", "exhausted"}:
            emit(RunEvaluationReportFailedEvent(gate_codes=retry_gate_codes))
            updated_edit_config = retry_decision.updated_edit_config
            head_adjustment = retry_decision.head_adjustment
            if head_adjustment is not None:
                emit(
                    RunAutoTuneAdjustmentEvent(
                        global_k=int(head_adjustment["global_k"]),
                        keep_low=int(head_adjustment["keep_low"]),
                        keep_high=int(head_adjustment["keep_high"]),
                    )
                )
            for diagnostic in retry_decision.diagnostics:
                emit_transition("retry_validation", diagnostic)
            if retry_disposition == "retry":
                next_attempt = retry_decision.next_attempt or (attempt + 1)
                return updated_edit_config, int(next_attempt), True
            emit(RunRetryExhaustedEvent(attempt=int(attempt)))
            return updated_edit_config, attempt, False
        emit(RunRetryValidationErrorEvent(summary=retry_summary))
        return edit_config, attempt, False
    if retry_controller:
        record_retry_attempt_fn(
            retry_controller,
            attempt=attempt,
            attempt_summary={
                "passed": True,
                "failures": [],
                "validation": {},
            },
            edit_config=edit_config,
        )
    return edit_config, attempt, False


def _process_attempt_result(
    *,
    attempt_state: _AttemptExecutionState,
    timings: dict[str, float],
    report_path_out: str | None,
    cfg: Any,
    profile_normalized: str,
    profile: str | None,
    baseline: str | None,
    edit_label: str | None,
    metric_kind: str | None,
    export_model_requested: bool,
    export_dir_override: str | None,
    telemetry: bool,
    resolved_loss_type: str,
    tokenizer: Any | None,
    tokenizer_hash: str | None,
    resolved_split: str,
    preview_count: int,
    final_count: int,
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
    guard_overhead_threshold: float,
    pm_acceptance_range: Any,
    pm_drift_band: Any,
    seed_bundle: dict[str, int | None],
    run_dir: Path,
    run_config: Any,
    auto_config: Any,
    resolved_device: Any,
    snapshot_provenance: dict[str, bool],
    edit_op: Any,
    adapter: Any,
    model: Any | None,
    measure_guard_overhead: bool,
    retry_controller: Any | None,
    validate_retry_evaluation_report_fn: Any,
    resolve_retry_validation_transition_fn: Any,
    record_retry_attempt_fn: Any,
    persist_run_report_outputs_fn: Any,
    assemble_run_report_fn: Any,
    cfg_value: Any,
    emit: RunEventEmitter,
    emit_diagnostic: Any,
    emit_guard_overhead_summary: Any,
    emit_transition: Any,
    halt: Any,
    fail_run: Any,
    optional_runtime_exceptions: tuple[type[BaseException], ...],
) -> _AttemptDecision:
    debug_metric_diffs_enabled = str(
        os.environ.get("DEBUG_METRIC_DIFFS", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    assembly_result = assemble_run_report_fn(
        core_report=attempt_state.core_report,
        cfg=cfg,
        run_context=run_config.context,
        profile_normalized=profile_normalized,
        auto_config=auto_config,
        resolved_device=resolved_device,
        seed_bundle=seed_bundle,
        guard_overhead_threshold=guard_overhead_threshold,
        model_profile=model_profile,
        determinism_meta=determinism_meta,
        pm_acceptance_range=pm_acceptance_range,
        pm_drift_band=pm_drift_band,
        tokenizer_hash=tokenizer_hash,
        resolved_split=resolved_split,
        preview_count=preview_count,
        final_count=final_count,
        snapshot_provenance=snapshot_provenance,
        edit_op=edit_op,
        edit_label=edit_label,
        run_dir=run_dir,
        run_config=run_config,
        resolved_loss_type=resolved_loss_type,
        timings=timings,
        guard_overhead_payload=attempt_state.guard_overhead_payload,
        baseline=baseline,
        preview_records=preview_records,
        final_records=final_records,
        use_mlm=use_mlm,
        preview_mask_counts=preview_mask_counts,
        final_mask_counts=final_mask_counts,
        profile=profile,
        used_fallback_split=used_fallback_split,
        baseline_report_data=baseline_report_data,
        effective_preview=effective_preview,
        effective_final=effective_final,
        metric_kind=metric_kind,
        window_plan=window_plan,
        debug_metric_diffs_enabled=debug_metric_diffs_enabled,
    )
    report = assembly_result.report
    timings = assembly_result.timings
    provenance_result = assembly_result.provenance_result
    metrics_enrichment = assembly_result.metrics_enrichment
    try:
        if provenance_result.missing_evaluation_windows_for_baseline:
            halt(
                "baseline_windows_missing",
                summary=(
                    provenance_result.missing_evaluation_windows_message
                    or "[INVARLOCK:E001] PAIRING-SCHEDULE-MISMATCH: baseline pairing requested but evaluation windows were not produced. Check capacity/pairing config."
                ),
            )
    except InvarlockError as exc:
        halt("invarlock_error", summary=str(exc), error=exc)
    except RuntimeError as exc:
        fail_run(str(exc), error=exc)
    _maybe_export_model_artifacts(
        cfg=cfg,
        run_dir=run_dir,
        report=report,
        adapter=adapter,
        model=model,
        tokenizer=tokenizer,
        export_model_requested=export_model_requested,
        export_dir_override=export_dir_override,
        cfg_value=cfg_value,
        emit_diagnostic=emit_diagnostic,
        optional_runtime_exceptions=optional_runtime_exceptions,
    )
    pairing_violations = metrics_enrichment.pairing_violations
    if pairing_violations:
        violation = pairing_violations[0]
        err = InvarlockError(
            code=violation.code,
            message=violation.message,
            details=violation.details,
        )
        halt("invarlock_error", summary=str(err), error=err)
    if metrics_enrichment.debug_diffs_line:
        emit_diagnostic(
            code="metric_diffs_debug",
            summary=metrics_enrichment.debug_diffs_line,
        )
    persistence_result = persist_run_report_outputs_fn(
        report=report,
        run_dir=run_dir,
        run_config=run_config,
        model=model,
        telemetry=telemetry,
    )
    report_path_out = persistence_result.report_path_out or report_path_out
    if persistence_result.telemetry_saved_path:
        emit(RunTelemetrySavedEvent(path=str(persistence_result.telemetry_saved_path)))
    elif persistence_result.telemetry_error:
        emit(RunTelemetryFailedEvent(error=str(persistence_result.telemetry_error)))
    _emit_primary_metric_summary_from_report(report=report, emit=emit)
    _enforce_guard_overhead_budget(
        report=report,
        run_config=run_config,
        measure_guard_overhead=measure_guard_overhead,
        guard_overhead_threshold=guard_overhead_threshold,
        emit_guard_overhead_summary=emit_guard_overhead_summary,
        halt=halt,
    )
    updated_edit_config, next_attempt, should_continue = _handle_retry_validation(
        retry_controller=retry_controller,
        baseline=baseline,
        report=report,
        baseline_report_data=baseline_report_data,
        edit_config=attempt_state.edit_config,
        attempt=attempt_state.attempt,
        validate_retry_evaluation_report_fn=validate_retry_evaluation_report_fn,
        resolve_retry_validation_transition_fn=resolve_retry_validation_transition_fn,
        record_retry_attempt_fn=record_retry_attempt_fn,
        emit=emit,
        emit_transition=emit_transition,
        emit_diagnostic=emit_diagnostic,
    )
    return _AttemptDecision(
        report=report,
        timings=dict(timings),
        report_path_out=report_path_out,
        edit_config=updated_edit_config,
        attempt=int(next_attempt),
        should_continue=should_continue,
    )


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
        execution_state.snapshot_provenance["restore_failed"] = True
        free_model_memory_fn(execution_state.model)
        execution_state.model = None
        execution_state.restore_fn = None
        emit_diagnostic(code="snapshot_restore_fallback", error=str(exc))
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
