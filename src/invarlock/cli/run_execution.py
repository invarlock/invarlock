"""CLI shell wrapper for config-driven run orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click
import typer

from invarlock.cli import run_config as run_config_mod
from invarlock.cli import run_pairing as run_pairing_mod
from invarlock.cli import run_warning_filters as run_warning_filters_mod
from invarlock.cli.output import (
    make_console,
    print_timing_summary,
    resolve_output_style,
    timed_step,
)
from invarlock.cli.run_masking import _apply_mlm_masks, _tokenizer_digest
from invarlock.cli.run_pairing_helpers import (
    _hash_sequences,
    _safe_int,
    _tensor_or_list_to_ints,
)
from invarlock.cli.run_runtime import (
    detect_model_profile,
    free_model_memory,
    get_psutil,
    get_torch,
    reset_optional_runtime_caches,
)
from invarlock.cli.run_runtime_exec import (
    SnapshotRestoreFailed as _SnapshotRestoreFailed,
)
from invarlock.cli.run_runtime_exec import (
    build_snapshot_execution_plan as _build_snapshot_execution_plan,
)
from invarlock.cli.run_runtime_exec import (
    execute_guarded_run as _execute_guarded_run,
)
from invarlock.cli.run_runtime_exec import (
    init_retry_controller as _init_retry_controller,
)
from invarlock.cli.run_runtime_exec import (
    load_model_with_cfg as _load_model_with_cfg,
)
from invarlock.cli.run_runtime_exec import run_bare_control as _run_bare_control
from invarlock.cli.run_serialization import _to_serialisable_dict
from invarlock.cli.run_shell_output import (
    _event,
    _format_guard_chain,
    _format_kv_line,
    _print_guard_overhead_summary,
    _print_pipeline_start,
    _print_retry_summary,
)
from invarlock.core.exit_codes import (
    resolve_command_exit_code as _resolve_exit_code,
)
from invarlock.core.retry import adjust_edit_params as _adjust_edit_params
from invarlock.core.run_baseline_evidence import (
    load_baseline_pairing_evidence as _load_baseline_pairing_evidence,
)
from invarlock.core.run_dataset_contract import (
    materialize_run_dataset as _materialize_run_dataset,
)
from invarlock.core.run_orchestrator import (
    RunExecutionAbort,
    RunExecutionHooks,
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
from invarlock.core.run_snapshot_contract import (
    resolve_snapshot_retry_transition as _resolve_snapshot_retry_transition_impl,
)
from invarlock.core.run_snapshot_policy import (
    resolve_snapshot_config as _resolve_snapshot_config_impl,
)
from invarlock.reporting.run_report_contract import (
    assemble_run_report as _assemble_run_report,
)
from invarlock.reporting.run_report_contract import (
    persist_run_report_outputs as _persist_run_report_outputs,
)
from invarlock.reporting.run_retry_validation import (
    validate_retry_evaluation_report as _validate_retry_evaluation_report,
)

if TYPE_CHECKING:
    from .config_execution import ConfigExecutionRequest

console = make_console()


def execute_config_run_request(request: ConfigExecutionRequest) -> str | None:
    reset_optional_runtime_caches()
    return execute_run_request(request)


def _build_run_execution_services() -> RunExecutionServices:
    return RunExecutionServices(
        SnapshotRestoreFailed=_SnapshotRestoreFailed,
        apply_mlm_masks=_apply_mlm_masks,
        adjust_edit_params=_adjust_edit_params,
        apply_warning_filters=run_warning_filters_mod._apply_warning_filters,
        assemble_run_report=_assemble_run_report,
        build_snapshot_execution_plan=_build_snapshot_execution_plan,
        build_provider_dataset_plan=_build_provider_dataset_plan,
        event=_event,
        execute_guarded_run=_execute_guarded_run,
        extract_pairing_schedule=run_pairing_mod.extract_pairing_schedule,
        load_baseline_pairing_evidence=_load_baseline_pairing_evidence,
        materialize_run_dataset=_materialize_run_dataset,
        format_guard_chain=_format_guard_chain,
        format_kv_line=_format_kv_line,
        free_model_memory=free_model_memory,
        hash_sequences=_hash_sequences,
        init_retry_controller=_init_retry_controller,
        load_model_with_cfg=_load_model_with_cfg,
        persist_run_report_outputs=_persist_run_report_outputs,
        prepare_config_for_run=run_config_mod.prepare_config_for_run,
        print_guard_overhead_summary=_print_guard_overhead_summary,
        print_pipeline_start=_print_pipeline_start,
        print_retry_summary=_print_retry_summary,
        resolve_device_and_output=run_config_mod.resolve_device_and_output,
        resolve_exit_code=_resolve_exit_code,
        resolve_pm_min_tokens_target=_resolve_pm_min_tokens_target,
        resolve_snapshot_config=_resolve_snapshot_config,
        resolve_snapshot_retry_transition=_resolve_snapshot_retry_transition_impl,
        run_bare_control=_run_bare_control,
        safe_int=_safe_int,
        tensor_or_list_to_ints=_tensor_or_list_to_ints,
        to_serialisable_dict=_to_serialisable_dict,
        tokenizer_digest=_tokenizer_digest,
        validate_retry_evaluation_report=_validate_retry_evaluation_report,
        validate_and_harvest_baseline_schedule=(
            run_pairing_mod.validate_and_harvest_baseline_schedule
        ),
        console=console,
        detect_model_profile=detect_model_profile,
        get_psutil=get_psutil,
        get_torch=get_torch,
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
    if not output_style.color:
        console.no_color = True
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


def execute_run_request(request: RunExecutionRequest) -> str | None:
    try:
        output_style = _resolve_shell_output_style(request)
        result = _execute_run_request_impl(
            request,
            services=_build_run_execution_services(),
            hooks=RunExecutionHooks(
                output_style=output_style,
                emit_line_fn=lambda line, markup=False: _emit_console_line(
                    line, markup=markup
                ),
                emit_blank_line_fn=_emit_console_blank_line,
                print_timing_summary_fn=print_timing_summary,
                timed_step_fn=timed_step,
                abort_exception_types=(typer.Exit, SystemExit, click.exceptions.Exit),
            ),
        )
        return result.report_path
    except RunExecutionAbort as exc:
        raise typer.Exit(exc.code) from exc
