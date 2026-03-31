"""CLI shell wrapper for config-driven run orchestration."""

from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Any

import typer

from invarlock.cli import overhead_utils as overhead_utils_mod
from invarlock.cli import run_artifact_output as run_artifact_output_mod
from invarlock.cli import run_artifacts as run_artifacts_mod
from invarlock.cli import run_config as run_config_mod
from invarlock.cli import run_execution_output as run_execution_output_mod
from invarlock.cli import run_pairing as run_pairing_mod
from invarlock.cli import run_runtime as run_runtime_mod
from invarlock.cli import run_runtime_exec as run_runtime_exec_mod
from invarlock.cli import run_warning_filters as run_warning_filters_mod
from invarlock.cli.output import (
    make_console,
    print_timing_summary,
)
from invarlock.cli.run_masking import _apply_mlm_masks, _tokenizer_digest
from invarlock.cli.run_overhead import plan_release_windows as _plan_release_windows
from invarlock.cli.run_pairing_helpers import (
    _hash_sequences,
    _safe_int,
    _tensor_or_list_to_ints,
)
from invarlock.cli.run_serialization import _to_serialisable_dict
from invarlock.cli.run_shell_output import _event
from invarlock.core import metric_provider_resolution as metric_provider_resolution_mod
from invarlock.core import provider_parity as provider_parity_mod
from invarlock.core import run_baseline_evidence as run_baseline_evidence_mod
from invarlock.core import (
    run_evaluation_windows_policy as run_evaluation_windows_policy_mod,
)
from invarlock.core import run_guard_overhead_policy as run_guard_overhead_policy_mod
from invarlock.core import run_report_payload_policy as run_report_payload_policy_mod
from invarlock.core.exceptions import ValidationError
from invarlock.core.retry import adjust_edit_params as _adjust_edit_params
from invarlock.core.run_dataset_contract import (
    materialize_run_dataset as _materialize_run_dataset,
)
from invarlock.core.run_execution_request_policy import (
    SupportsRunExecutionRequest,
)
from invarlock.core.run_execution_request_policy import (
    build_run_execution_request as _build_run_execution_request_impl,
)
from invarlock.core.run_orchestrator import (
    RunExecutionEvent,
    RunExecutionRequest,
    RunExecutionServices,
)
from invarlock.core.run_orchestrator import (
    execute_run_request as _execute_run_request_impl,
)
from invarlock.core.run_policy import (
    resolve_pm_min_tokens_target as _resolve_pm_min_tokens_target,
)
from invarlock.core.run_provider_dataset_plan import (
    build_provider_dataset_plan as _build_provider_dataset_plan,
)
from invarlock.core.run_retry_policy import (
    build_retry_result_summary as _build_retry_result_summary_impl,
)
from invarlock.core.run_snapshot_contract import (
    resolve_snapshot_retry_transition as _resolve_snapshot_retry_transition_impl,
)
from invarlock.core.run_snapshot_policy import (
    resolve_snapshot_config as _resolve_snapshot_config_impl,
)
from invarlock.eval import data as eval_data_mod
from invarlock.eval import window_planning as window_planning_mod
from invarlock.exit_codes import (
    resolve_command_exit_code as _resolve_exit_code,
)
from invarlock.reporting import report_make as report_make_mod
from invarlock.reporting import report_telemetry as report_telemetry_mod
from invarlock.reporting import report_types as report_types_mod
from invarlock.reporting import run_provenance_contract as run_provenance_contract_mod
from invarlock.reporting import run_report_contract as run_report_contract_mod
from invarlock.reporting import (
    run_report_metrics_contract as run_report_metrics_contract_mod,
)
from invarlock.reporting import telemetry as telemetry_mod
from invarlock.reporting.run_retry_validation import (
    validate_retry_evaluation_report as _validate_retry_evaluation_report,
)

if TYPE_CHECKING:
    from .config_execution import ConfigExecutionRequest

console = make_console()


def execute_config_run_request(request: ConfigExecutionRequest) -> str | None:
    run_runtime_mod.reset_optional_runtime_caches()
    return execute_run_request(request)


def _build_run_execution_services() -> RunExecutionServices:
    return RunExecutionServices(
        SnapshotRestoreFailed=run_runtime_exec_mod.SnapshotRestoreFailed,
        adjust_edit_params=_adjust_edit_params,
        assemble_run_report=_assemble_run_report_with_runtime_deps,
        build_snapshot_execution_plan=run_runtime_exec_mod.build_snapshot_execution_plan,
        build_provider_dataset_plan=_build_provider_dataset_plan_with_runtime_deps,
        execute_guarded_run=run_runtime_exec_mod.execute_guarded_run,
        load_baseline_pairing_evidence=_load_baseline_pairing_evidence_with_runtime_deps,
        materialize_run_dataset=_materialize_run_dataset_with_runtime_deps,
        free_model_memory=run_runtime_mod.free_model_memory,
        init_retry_controller=run_runtime_exec_mod.init_retry_controller,
        load_model_with_cfg=run_runtime_exec_mod.load_model_with_cfg,
        persist_run_report_outputs=_persist_run_report_outputs_with_runtime_deps,
        prepare_config_for_run=_prepare_config_for_run_with_runtime_deps,
        resolve_device_and_output=_resolve_device_and_output_with_runtime_deps,
        resolve_snapshot_config=_resolve_snapshot_config,
        resolve_snapshot_retry_transition=_resolve_snapshot_retry_transition_impl,
        run_bare_control=run_runtime_exec_mod.run_bare_control,
        safe_int=_safe_int,
        to_serialisable_dict=_to_serialisable_dict,
        validate_retry_evaluation_report=(
            _validate_retry_evaluation_report_with_runtime_deps
        ),
        validate_and_harvest_baseline_schedule=partial(
            run_pairing_mod.validate_and_harvest_baseline_schedule,
            typed_failures=True,
        ),
        materialize_baseline_pairing_schedule=(
            _materialize_baseline_pairing_schedule_with_runtime_deps
        ),
        resolve_tokenizer=run_runtime_mod.resolve_tokenizer,
        detect_model_profile=run_runtime_mod.detect_model_profile,
        get_psutil=run_runtime_mod.get_psutil,
        get_torch=run_runtime_mod.get_torch,
    )


def _prepare_config_for_run_with_runtime_deps(**kwargs: Any) -> Any:
    kwargs.pop("console", None)
    return run_config_mod.prepare_config_for_run(**kwargs)


def _resolve_device_and_output_with_runtime_deps(*args: Any, **kwargs: Any) -> Any:
    kwargs.pop("console", None)
    return run_config_mod.resolve_device_and_output(*args, **kwargs)


def _resolve_provider_and_split_for_dataset_plan(*args: Any, **kwargs: Any) -> Any:
    kwargs.pop("console", None)
    return run_config_mod.resolve_provider_and_split(*args, **kwargs)


def _load_baseline_pairing_evidence_with_runtime_deps(**kwargs: Any) -> Any:
    return run_baseline_evidence_mod.load_baseline_pairing_evidence(
        **kwargs,
        extract_pairing_schedule_fn=run_pairing_mod.extract_pairing_schedule,
    )


def _materialize_run_dataset_with_runtime_deps(**kwargs: Any) -> Any:
    kwargs.pop("console", None)
    return _materialize_run_dataset(**kwargs)


def _materialize_baseline_pairing_schedule_with_runtime_deps(
    **kwargs: Any,
) -> Any:
    return run_baseline_evidence_mod.materialize_baseline_pairing_schedule(
        **kwargs,
        apply_mlm_masks_fn=_apply_mlm_masks,
        resolve_pm_min_tokens_target_fn=_resolve_pm_min_tokens_target,
        hash_sequences_fn=_hash_sequences,
        tensor_or_list_to_ints_fn=_tensor_or_list_to_ints,
    )


def _build_provider_dataset_plan_with_runtime_deps(**kwargs: Any) -> Any:
    return _build_provider_dataset_plan(
        **kwargs,
        get_provider_fn=eval_data_mod.get_provider,
        resolve_provider_and_split_fn=_resolve_provider_and_split_for_dataset_plan,
        resolve_tokenizer_fn=run_runtime_mod.resolve_tokenizer,
        maybe_plan_release_windows_fn=_plan_release_windows,
        resolve_effective_windows_fn=window_planning_mod.resolve_effective_windows,
        apply_mlm_masks_fn=_apply_mlm_masks,
        resolve_pm_min_tokens_target_fn=_resolve_pm_min_tokens_target,
        hash_sequences_fn=_hash_sequences,
        tokenizer_digest_fn=_tokenizer_digest,
        safe_int_fn=_safe_int,
        tensor_or_list_to_ints_fn=_tensor_or_list_to_ints,
    )


def _assemble_run_report_with_runtime_deps(**kwargs: Any) -> Any:
    return run_report_contract_mod.assemble_run_report(
        **kwargs,
        create_empty_report_fn=report_types_mod.create_empty_report,
        build_run_report_context_fn=(
            run_report_payload_policy_mod.build_run_report_context
        ),
        build_run_report_meta_fn=run_report_payload_policy_mod.build_run_report_meta,
        canonical_dataset_id_fn=run_pairing_mod._canonical_dataset_id,
        safe_int_fn=_safe_int,
        build_run_report_data_fn=run_report_payload_policy_mod.build_run_report_data,
        build_snapshot_provenance_fn=(
            run_report_payload_policy_mod.build_snapshot_provenance
        ),
        build_edit_payload_fn=run_report_payload_policy_mod.build_edit_payload,
        persist_ref_masks_fn=run_artifacts_mod.persist_ref_masks,
        build_artifacts_payload_fn=(
            run_report_payload_policy_mod.build_artifacts_payload
        ),
        merge_core_timing_metrics_fn=(
            run_report_payload_policy_mod.merge_core_timing_metrics
        ),
        build_metrics_payload_fn=run_report_payload_policy_mod.build_metrics_payload,
        prepare_guard_overhead_report_fn=(
            _prepare_guard_overhead_report_with_runtime_deps
        ),
        finalize_run_provenance_fn=(_finalize_run_provenance_with_runtime_deps),
        build_guard_entries_fn=run_report_payload_policy_mod.build_guard_entries,
        build_flags_payload_fn=run_report_payload_policy_mod.build_flags_payload,
        enrich_run_report_metrics_fn=(_enrich_run_report_metrics_with_runtime_deps),
        optional_torch_fn=run_runtime_mod.get_torch,
        environ=os.environ,
    )


def _persist_run_report_outputs_with_runtime_deps(**kwargs: Any) -> Any:
    kwargs.pop("console", None)
    report = kwargs["report"]
    run_dir = kwargs["run_dir"]
    run_config = kwargs["run_config"]
    persistence_result = run_report_contract_mod.persist_run_report_outputs(
        **kwargs,
        save_telemetry_report_fn=telemetry_mod.save_telemetry_report,
    )
    run_artifact_output_mod.postprocess_and_summarize(
        report=report,
        run_dir=run_dir,
        run_config=run_config,
        console=console,
        saved_files=persistence_result.saved_files,
    )
    return persistence_result


def _prepare_guard_overhead_report_with_runtime_deps(*args: Any, **kwargs: Any) -> Any:
    return run_guard_overhead_policy_mod.prepare_guard_overhead_report(
        *args,
        **kwargs,
        extract_pm_snapshot_for_overhead_fn=(
            overhead_utils_mod._extract_pm_snapshot_for_overhead
        ),
        validate_guard_overhead_fn=run_runtime_mod.validate_guard_overhead,
    )


def _finalize_run_provenance_with_runtime_deps(**kwargs: Any) -> Any:
    return run_provenance_contract_mod.finalize_run_provenance(
        **kwargs,
        serialize_evaluation_windows_fn=(
            run_evaluation_windows_policy_mod.serialize_evaluation_windows
        ),
        build_fallback_evaluation_windows_fn=(
            run_evaluation_windows_policy_mod.build_fallback_evaluation_windows
        ),
        compute_provider_digest_fn=run_pairing_mod.compute_provider_digest,
        enforce_provider_parity_fn=provider_parity_mod.enforce_provider_parity,
    )


def _enrich_run_report_metrics_with_runtime_deps(**kwargs: Any) -> Any:
    return run_report_metrics_contract_mod.enrich_run_report_metrics(
        **kwargs,
        resolve_metric_and_provider_fn=(
            metric_provider_resolution_mod.resolve_metric_and_provider
        ),
    )


def _validate_retry_evaluation_report_with_runtime_deps(**kwargs: Any) -> Any:
    return _validate_retry_evaluation_report(
        **kwargs,
        build_retry_result_summary_fn=_build_retry_result_summary_impl,
        make_report_fn=report_make_mod.make_report,
        telemetry_output_enabled_fn=report_telemetry_mod.telemetry_output_enabled,
        telemetry_summary_line_fn=report_telemetry_mod.telemetry_summary_line,
    )


def _resolve_snapshot_config(context: object | None) -> dict[str, Any]:
    return _resolve_snapshot_config_impl(
        context,
        to_serialisable_dict_fn=_to_serialisable_dict,
    )


def _resolve_shell_output_style(request: Any) -> Any:
    return run_execution_output_mod.resolve_shell_output_style(console, request)


def _emit_console_line(line: str, *, markup: bool = False) -> None:
    run_execution_output_mod.emit_console_line(console, line, markup=markup)


def _emit_console_blank_line() -> None:
    run_execution_output_mod.emit_console_blank_line(console)


def _begin_progress_step(key: str) -> None:
    run_execution_output_mod.begin_progress_step(console, key)


def _complete_progress_step(
    key: str,
    *,
    tag: str,
    message: str,
    emoji: str | None = None,
) -> None:
    run_execution_output_mod.complete_progress_step(
        console,
        key,
        tag=tag,
        message=message,
        emoji=emoji,
    )


def _transition_progress_step(
    from_key: str,
    *,
    from_tag: str,
    from_message: str,
    to_key: str,
    from_emoji: str | None = None,
) -> None:
    run_execution_output_mod.transition_progress_step(
        console,
        from_key,
        from_tag=from_tag,
        from_message=from_message,
        to_key=to_key,
        from_emoji=from_emoji,
    )


def _render_run_execution_event(event: RunExecutionEvent) -> None:
    run_execution_output_mod.render_run_execution_event(console, event)


def _to_core_run_execution_request(
    request: SupportsRunExecutionRequest,
) -> RunExecutionRequest:
    return _build_core_run_execution_request(request)


def _build_core_run_execution_request(
    request: SupportsRunExecutionRequest,
) -> RunExecutionRequest:
    return _build_run_execution_request_impl(request)


def _exit_code_for_failure(failure: Any, *, profile: str | None) -> int:
    if failure is None:
        return 1
    code = str(getattr(failure, "code", "") or "")
    message = str(
        getattr(failure, "message", "")
        or getattr(getattr(failure, "error", None), "message", "")
        or ""
    )
    if code == "baseline_windows_missing":
        return 3
    if code in {"unknown_edit", "schema_invalid_run_report"}:
        return 2
    error = getattr(failure, "error", None)
    if isinstance(error, ValidationError) and (
        "Invalid tier" in message
        or "Invalid probes" in message
        or "Device validation failed" in message
    ):
        return 1
    if isinstance(error, Exception):
        return _resolve_exit_code(error, profile=profile)
    if code in {"torch_missing", "config_file_missing", "edit_name_missing"}:
        return 1
    return 1


def execute_run_request(request: Any) -> str | None:
    _resolve_shell_output_style(request)
    console._invarlock_progress_steps = {}
    console._invarlock_progress_completed = set()
    run_warning_filters_mod._apply_warning_filters(
        str(request.profile or "").strip().lower() or None
    )
    core_request = _to_core_run_execution_request(request)
    outcome = _execute_run_request_impl(
        core_request,
        services=_build_run_execution_services(),
        observer=_render_run_execution_event,
    )
    _complete_progress_step(
        "execute",
        tag="EXEC",
        message="Execute pipeline",
        emoji="⚙️",
    )
    if not outcome.ok or outcome.result is None:
        raise typer.Exit(
            _exit_code_for_failure(
                outcome.failure,
                profile=str(getattr(request, "profile", None) or None),
            )
        )
    output_style = getattr(console, "_invarlock_output_style", None)
    if bool(getattr(request, "progress", False)) and output_style is not None:
        completed_steps = getattr(console, "_invarlock_progress_completed", set())
        progress_fallback_steps = (
            ("load_model", "INIT", "Loading model", "🔧"),
            ("execute", "EXEC", "Execute pipeline", "⚙️"),
        )
        for key, tag, message, emoji in progress_fallback_steps:
            if key in completed_steps:
                continue
            elapsed = outcome.result.timings.get(key)
            if isinstance(elapsed, int | float):
                _event(console, tag, f"{message} done ({elapsed:.2f}s)", emoji=emoji)
    if bool(getattr(request, "timing", False)) and output_style is not None:
        timing_summary = outcome.result.timing_summary
        order = [
            ("Load model", "load_model"),
            ("Load dataset", "load_dataset"),
            ("Execute", "execute"),
            ("Total", "total"),
        ]
        extra_lines: list[str] = []
        if timing_summary is not None:
            label_map = {
                "load_model": "Load model",
                "load_dataset": "Load dataset",
                "prepare": "Prepare",
                "prepare_guards": "Prep guards",
                "edit": "Edit",
                "guards": "Guards",
                "eval": "Eval",
                "finalize": "Finalize",
                "execute": "Execute",
                "total": "Total",
            }
            order = [
                (label_map.get(key, key.replace("_", " ").title()), key)
                for key in timing_summary.ordered_keys
            ]
            if timing_summary.memory_mb_peak is not None:
                extra_lines.append(
                    f"  Peak memory: {timing_summary.memory_mb_peak:.1f} MB"
                )
            if timing_summary.gpu_memory_mb_peak is not None:
                extra_lines.append(
                    f"  Peak GPU mem: {timing_summary.gpu_memory_mb_peak:.1f} MB"
                )
        print_timing_summary(
            console=console,
            timings=dict(outcome.result.timings),
            style=output_style,
            order=order,
            extra_lines=extra_lines,
        )
    return outcome.result.report_path if outcome.result is not None else None
