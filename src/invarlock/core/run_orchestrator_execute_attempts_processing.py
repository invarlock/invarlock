"""Result-processing helpers for run orchestrator attempt execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from invarlock.core.exceptions import InvarlockError
from invarlock.core.run_orchestrator_execute_attempts_emit import (
    _emit_primary_metric_summary_from_report,
)
from invarlock.core.run_orchestrator_execute_attempts_export import (
    _maybe_export_model_artifacts,
)
from invarlock.core.run_orchestrator_execute_helpers import (
    RunEventEmitter,
    _AttemptDecision,
    _AttemptExecutionState,
)
from invarlock.core.run_orchestrator_types import (
    RunAutoTuneAdjustmentEvent,
    RunEvaluationReportFailedEvent,
    RunEvaluationReportPassedEvent,
    RunEvaluationReportStartedEvent,
    RunRetryExhaustedEvent,
    RunRetryValidationErrorEvent,
    RunTelemetryFailedEvent,
    RunTelemetrySavedEvent,
)


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


__all__ = [
    "_enforce_guard_overhead_budget",
    "_handle_retry_validation",
    "_process_attempt_result",
]
