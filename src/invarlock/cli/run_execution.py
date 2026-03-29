"""CLI shell wrapper for config-driven run orchestration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import typer

from invarlock.cli import overhead_utils as overhead_utils_mod
from invarlock.cli import run_artifact_output as run_artifact_output_mod
from invarlock.cli import run_artifacts as run_artifacts_mod
from invarlock.cli import run_config as run_config_mod
from invarlock.cli import run_pairing as run_pairing_mod
from invarlock.cli import run_runtime as run_runtime_mod
from invarlock.cli import run_runtime_exec as run_runtime_exec_mod
from invarlock.cli import run_warning_filters as run_warning_filters_mod
from invarlock.cli.output import (
    make_console,
    print_timing_summary,
    resolve_output_style,
    timed_step,
)
from invarlock.cli.run_masking import _apply_mlm_masks, _tokenizer_digest
from invarlock.cli.run_overhead import plan_release_windows as _plan_release_windows
from invarlock.cli.run_pairing_helpers import (
    _hash_sequences,
    _safe_int,
    _tensor_or_list_to_ints,
)
from invarlock.cli.run_serialization import _to_serialisable_dict
from invarlock.cli.run_shell_output import (
    _device_resolution_note,
    _event,
    _format_kv_line,
    _print_guard_overhead_summary,
)
from invarlock.cli.run_shell_output import (
    _print_retry_summary as _shell_print_retry_summary,
)
from invarlock.core import metric_provider_resolution as metric_provider_resolution_mod
from invarlock.core import provider_parity as provider_parity_mod
from invarlock.core import run_baseline_evidence as run_baseline_evidence_mod
from invarlock.core import (
    run_evaluation_windows_policy as run_evaluation_windows_policy_mod,
)
from invarlock.core import run_guard_overhead_policy as run_guard_overhead_policy_mod
from invarlock.core import run_report_payload_policy as run_report_payload_policy_mod
from invarlock.core.exceptions import ValidationError
from invarlock.core.exit_codes import (
    resolve_command_exit_code as _resolve_exit_code,
)
from invarlock.core.retry import adjust_edit_params as _adjust_edit_params
from invarlock.core.run_dataset_contract import (
    materialize_run_dataset as _materialize_run_dataset,
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


def _print_retry_summary(_console: Any, retry_controller: Any | None) -> None:
    """Compatibility shim retained for tests that patch the retry-summary seam."""

    _shell_print_retry_summary(_console, retry_controller)


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
        execute_guarded_run=_execute_guarded_run_with_runtime_deps,
        load_baseline_pairing_evidence=_load_baseline_pairing_evidence_with_runtime_deps,
        materialize_run_dataset=_materialize_run_dataset_with_runtime_deps,
        free_model_memory=run_runtime_mod.free_model_memory,
        init_retry_controller=_init_retry_controller_with_runtime_deps,
        load_model_with_cfg=_load_model_with_cfg_with_runtime_deps,
        persist_run_report_outputs=_persist_run_report_outputs_with_runtime_deps,
        prepare_config_for_run=_prepare_config_for_run_with_runtime_deps,
        resolve_device_and_output=_resolve_device_and_output_with_runtime_deps,
        resolve_snapshot_config=_resolve_snapshot_config,
        resolve_snapshot_retry_transition=_resolve_snapshot_retry_transition_impl,
        run_bare_control=_run_bare_control_with_runtime_deps,
        safe_int=_safe_int,
        to_serialisable_dict=_to_serialisable_dict,
        validate_retry_evaluation_report=(
            _validate_retry_evaluation_report_with_runtime_deps
        ),
        validate_and_harvest_baseline_schedule=(
            run_pairing_mod.validate_and_harvest_baseline_schedule
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


def _init_retry_controller_with_runtime_deps(**kwargs: Any) -> Any:
    kwargs.pop("console", None)
    return run_runtime_exec_mod.init_retry_controller(
        **kwargs,
        console=console,
    )


def _run_bare_control_with_runtime_deps(**kwargs: Any) -> Any:
    kwargs.pop("console", None)
    return run_runtime_exec_mod.run_bare_control(
        **kwargs,
        console=console,
    )


def _execute_guarded_run_with_runtime_deps(**kwargs: Any) -> Any:
    kwargs.pop("console", None)
    style = getattr(console, "_invarlock_output_style", None)
    if style is None:
        return run_runtime_exec_mod.execute_guarded_run(
            **kwargs,
            console=console,
        )
    with timed_step(
        console=console,
        style=style,
        timings=None,
        key="execute",
        tag="EXEC",
        message="Execute pipeline",
        emoji="⚙️",
    ):
        return run_runtime_exec_mod.execute_guarded_run(
            **kwargs,
            console=console,
        )


def _load_model_with_cfg_with_runtime_deps(*args: Any, **kwargs: Any) -> Any:
    style = getattr(console, "_invarlock_output_style", None)
    if style is None:
        return run_runtime_exec_mod.load_model_with_cfg(*args, **kwargs)
    with timed_step(
        console=console,
        style=style,
        timings=None,
        key="load_model",
        tag="INIT",
        message="Loading model",
        emoji="🔧",
    ):
        return run_runtime_exec_mod.load_model_with_cfg(*args, **kwargs)


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
    output_style = resolve_output_style(
        style=str(request.style) if request.style is not None else None,
        profile=str(request.profile) if request.profile is not None else None,
        progress=bool(request.progress),
        timing=bool(request.timing),
        no_color=bool(request.no_color),
    )
    console._invarlock_output_style = output_style
    console.no_color = not bool(output_style.color)
    return output_style


def _emit_console_line(line: str, *, markup: bool = False) -> None:
    if markup:
        console.print(line)
        return
    try:
        console.print(line, markup=False)
    except TypeError:
        console.print(line)


def _emit_console_blank_line() -> None:
    console.print("")


def _render_run_execution_event(event: RunExecutionEvent) -> None:
    phase = event.phase
    name = event.code
    payload = event.details

    if phase == "metadata":
        if name == "device_resolved":
            resolved_device = str(payload.get("resolved_device", ""))
            requested_device = str(payload.get("requested_device") or "auto")
            resolution_note = _device_resolution_note(
                requested_device,
                resolved_device,
            )
            _emit_console_line(
                _format_kv_line("Device", f"{resolved_device} ({resolution_note})"),
                markup=False,
            )
            return
        if name == "run_directory_ready":
            _emit_console_line(
                _format_kv_line("Output", str(payload.get("run_dir", ""))),
                markup=False,
            )
            _emit_console_line(
                _format_kv_line("Run ID", str(payload.get("run_id", ""))),
                markup=False,
            )
            return
        if name == "edit_selected":
            _emit_console_line(
                _format_kv_line("Edit", str(payload.get("edit_name", ""))),
                markup=False,
            )
            return
        if name == "guard_chain_resolved":
            guard_names = [str(item) for item in payload.get("guard_names", [])]
            _emit_console_line(
                _format_kv_line("Guards", " → ".join(guard_names)),
                markup=False,
            )
            return

    if phase == "diagnostic":
        if name == "guard_missing":
            _event(
                console,
                "WARN",
                f"Guard '{payload.get('guard_name', '')}' not found, skipping",
                emoji="⚠️",
            )
            return
        if name == "export_tokenizer_missing":
            _event(
                console,
                "WARN",
                "Exported model checkpoint without tokenizer artifacts; local tokenizer reload may fail.",
                emoji="⚠️",
            )
            return
        if name == "export_adapter_directory_missing":
            _event(
                console,
                "WARN",
                "Model export requested but adapter did not save a HF directory.",
                emoji="⚠️",
            )
            return
        if name == "export_failed":
            _event(
                console,
                "WARN",
                "Model export requested but failed due to an unexpected error.",
                emoji="⚠️",
            )
            return
        if name == "snapshot_restore_fallback":
            _event(
                console,
                "WARN",
                "Snapshot restore failed; switching to reload-per-attempt.",
                emoji="⚠️",
            )
            error = payload.get("error")
            if error:
                _event(console, "WARN", f"↳ {error}")
            return
        if name == "retry_validation_telemetry_summary":
            _emit_console_line(str(payload.get("summary", "")), markup=False)
            return
        if name == "metric_diffs_debug":
            _emit_console_line(
                f"[debug] DEBUG_METRIC_DIFFS: {payload.get('summary', '')}",
                markup=False,
            )
            return

        message = payload.get("message")
        if isinstance(message, str) and message:
            tag = str(payload.get("tag") or payload.get("severity") or "INFO").upper()
            emoji = payload.get("emoji")
            if not isinstance(emoji, str):
                emoji = None
            _event(console, tag, message, emoji=emoji)
            return

    if phase == "summary" and name == "guard_overhead_summary":
        _print_guard_overhead_summary(
            console,
            payload.get("guard_overhead_info") or {},
            default_threshold=float(payload.get("default_threshold", 0.01) or 0.01),
        )
        return

    if phase == "summary" and name == "retry_summary":
        summary = payload.get("summary")
        if isinstance(summary, dict):
            _emit_console_blank_line()
            _event(
                console,
                "METRIC",
                f"Retry Summary: {summary.get('total_attempts', 0)} attempts in {float(summary.get('elapsed_time', 0.0) or 0.0):.1f}s",
                emoji="📊",
            )
        return

    if phase != "status":
        return

    def _emit_status_line(tag: str, message: str, *, emoji: str | None = None) -> None:
        _event(console, tag, message, emoji=emoji)

    if name == "torch_missing":
        _emit_status_line(
            "FAIL",
            'Torch is required for this command. Install extras with: pip install "invarlock[hf]" or "invarlock[adapters]".',
            emoji="❌",
        )
        return
    if name == "deterministic_seed_bundle":
        torch_seed = payload.get("torch_seed")
        torch_display = torch_seed if torch_seed is not None else "N/A"
        _emit_status_line(
            "INIT",
            "Deterministic seeds → "
            f"python={payload.get('python_seed')}, numpy={payload.get('numpy_seed')}, torch={torch_display}",
            emoji="🎲",
        )
        return
    if name == "baseline_schedule_loaded":
        _emit_status_line(
            "DATA",
            "Loaded baseline evaluation schedule for pairing",
            emoji="🧬",
        )
        return
    if name == "pipeline_start":
        _emit_status_line("INIT", "Starting InvarLock pipeline...", emoji="🚀")
        return
    if name == "config_loading":
        _emit_status_line(
            "INIT",
            f"Loading configuration: {payload.get('config_path', '')}",
            emoji="📋",
        )
        return
    if name == "config_loaded":
        return
    if name == "edit_name_missing":
        _emit_status_line(
            "FAIL",
            "Edit configuration must specify a non-empty `edit.name`.",
            emoji="❌",
        )
        return
    if name == "unknown_edit":
        _emit_status_line(
            "FAIL",
            f"Unknown edit '{payload.get('edit_name', '')}'.",
            emoji="❌",
        )
        return
    if name == "guard_missing":
        _emit_status_line(
            "WARN",
            f"Guard '{payload.get('guard_name', '')}' not found, skipping",
            emoji="⚠️",
        )
        return
    if name == "adapter_selected":
        _emit_status_line(
            "DATA",
            f"Adapter: {payload.get('adapter_name', '')}",
            emoji="🔌",
        )
        return
    if name == "dataset_loading":
        _emit_status_line(
            "DATA",
            f"Loading dataset: {payload.get('provider', '')}",
            emoji="📊",
        )
        return
    if name == "debug_calibration_batch_sizes":
        _emit_console_line(
            "[debug] calibration batch size => preview="
            f"{payload.get('preview_count')} final={payload.get('final_count')} total={payload.get('total_count')}",
            markup=False,
        )
        return
    if name == "debug_masked_tokens":
        _emit_console_line(
            f"[debug] masked tokens (preview/final) = {payload.get('preview_masked')}/{payload.get('final_masked')}",
            markup=False,
        )
        return
    if name == "debug_preview_labels":
        _emit_console_line(
            f"[debug] sample labels first preview entry (first 10) = {payload.get('labels', [])}",
            markup=False,
        )
        return
    if name == "execute_pipeline":
        _emit_status_line(
            "EXEC",
            f"Executing pipeline with {payload.get('guard_count', 0)} guards...",
            emoji="⚙️",
        )
        return
    if name == "load_model_once":
        _emit_status_line(
            "INIT",
            f"Loading model once: {payload.get('model_id', '')}",
            emoji="🔧",
        )
        return
    if name == "snapshot_mode":
        state = "enabled" if bool(payload.get("enabled")) else "disabled"
        _emit_status_line("INIT", f"Snapshot mode: {state}", emoji="💾")
        return
    if name == "attempt_started":
        max_attempts = payload.get("max_attempts")
        message = (
            f"Attempt {payload.get('attempt')}/{max_attempts}"
            if max_attempts is not None
            else f"Attempt {payload.get('attempt')}"
        )
        _emit_status_line("EXEC", message, emoji="🚀")
        return
    if name == "retry_attempt_started":
        _emit_status_line(
            "EXEC",
            f"Retry attempt {payload.get('attempt')}/{payload.get('max_attempts')}",
            emoji="🔄",
        )
        return
    if name == "baseline_windows_missing":
        _emit_status_line("FAIL", str(payload.get("message", "")), emoji="❌")
        return
    if name == "invarlock_error":
        _emit_status_line("FAIL", str(payload.get("message", "")), emoji="❌")
        return
    if name == "telemetry_saved":
        _emit_status_line(
            "DATA",
            f"Telemetry: {payload.get('path', '')}",
            emoji="📈",
        )
        return
    if name == "telemetry_failed":
        _emit_status_line(
            "WARN",
            f"Telemetry export failed: {payload.get('error', '')}",
            emoji="⚠️",
        )
        return
    if name == "primary_metric_summary":
        _emit_status_line(
            "METRIC",
            f"Primary Metric [{payload.get('metric_kind', 'primary')}] — preview: {float(payload.get('preview', 0.0)):.3f}, final: {float(payload.get('final', 0.0)):.3f}",
            emoji="📌",
        )
        return
    if name == "baseline_ratio":
        _emit_status_line(
            "METRIC",
            f"Ratio vs baseline [{payload.get('metric_kind', 'primary')}]: {float(payload.get('ratio', 0.0)):.3f}",
            emoji="🔗",
        )
        return
    if name == "guard_overhead_gate_failed":
        _emit_status_line(
            "FAIL",
            "Guard overhead gate FAILED: Guards add more than the permitted budget",
            emoji="⚠️",
        )
        return
    if name == "guard_overhead_budget_exceeded":
        threshold_fraction = float(payload.get("threshold_fraction", 0.01) or 0.01)
        _emit_status_line(
            "FAIL",
            "Guard overhead gate exceeded the configured budget "
            f"(>{threshold_fraction * 100:.1f}% increase)",
            emoji="❌",
        )
        return
    if name == "evaluation_report_started":
        _emit_status_line("EXEC", "Generating evaluation report...", emoji="📜")
        return
    if name == "evaluation_report_passed":
        _emit_status_line("PASS", "Evaluation report PASSED all gates!", emoji="✅")
        return
    if name == "evaluation_report_failed":
        _emit_status_line(
            "FAIL",
            "Evaluation report FAILED gates: "
            + ", ".join(str(item) for item in payload.get("failed_gates", [])),
            emoji="⚠️",
        )
        return
    if name == "auto_tune_adjustment":
        _emit_status_line(
            "INIT",
            "Auto-tune adjust: global_k → "
            f"{payload.get('global_k')} "
            f"(bounds {payload.get('keep_low')}-{payload.get('keep_high')})",
            emoji="🔧",
        )
        return
    if name == "retry_exhausted":
        _emit_status_line(
            "FAIL",
            f"Exhausted retry budget after {payload.get('attempt')} attempts",
            emoji="❌",
        )
        return
    if name == "retry_validation_error":
        _emit_status_line(
            "WARN",
            f"Evaluation report validation failed: {payload.get('message', '')}",
            emoji="⚠️",
        )
        return
    if name == "config_file_missing":
        _emit_status_line(
            "FAIL",
            f"Configuration file not found: {payload.get('path', '')}",
            emoji="❌",
        )
        return
    if name == "schema_invalid_run_report":
        _emit_status_line(
            "FAIL",
            "Schema invalid: run report structure failed validation",
            emoji="❌",
        )
        return
    if name == "pipeline_failed":
        _emit_status_line(
            "FAIL",
            f"Pipeline execution failed: {payload.get('error', '')}",
            emoji="❌",
        )
        return
    if name == "cleanup_status":
        status = "removed" if bool(payload.get("removed")) else "skipped"
        _emit_status_line("INFO", f"Cleanup: {status}", emoji="🧹")
        return


def _to_core_run_execution_request(request: Any) -> RunExecutionRequest:
    return RunExecutionRequest(
        config=request.config,
        device=getattr(request, "device", None),
        profile=getattr(request, "profile", None),
        out=getattr(request, "out", None),
        edit=getattr(request, "edit", None),
        edit_label=getattr(request, "edit_label", None),
        tier=getattr(request, "tier", None),
        metric_kind=getattr(request, "metric_kind", None),
        probes=getattr(request, "probes", None),
        until_pass=bool(getattr(request, "until_pass", False)),
        max_attempts=int(getattr(request, "max_attempts", 3)),
        timeout=getattr(request, "timeout", None),
        baseline=getattr(request, "baseline", None),
        no_cleanup=bool(getattr(request, "no_cleanup", False)),
        capture_timings=bool(getattr(request, "timing", False)),
        telemetry=bool(getattr(request, "telemetry", False)),
        prefer_local_files_only=bool(
            getattr(request, "prefer_local_files_only", False)
        ),
    )


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
    run_warning_filters_mod._apply_warning_filters(
        str(request.profile or "").strip().lower() or None
    )
    core_request = _to_core_run_execution_request(request)
    outcome = _execute_run_request_impl(
        core_request,
        services=_build_run_execution_services(),
        observer=_render_run_execution_event,
    )
    if not outcome.ok or outcome.result is None:
        raise typer.Exit(
            _exit_code_for_failure(
                outcome.failure,
                profile=str(getattr(request, "profile", None) or None),
            )
        )
    output_style = getattr(console, "_invarlock_output_style", None)
    if bool(getattr(request, "timing", False)) and output_style is not None:
        print_timing_summary(
            console=console,
            timings=dict(outcome.result.timings),
            style=output_style,
            order=[
                ("Load model", "load_model"),
                ("Load dataset", "load_dataset"),
                ("Execute", "execute"),
                ("Total", "total"),
            ],
            extra_lines=[],
        )
    return outcome.result.report_path if outcome.result is not None else None
