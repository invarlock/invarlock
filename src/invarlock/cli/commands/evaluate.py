"""
InvarLock CLI Evaluate Command
=========================

Hero path: Compare & Evaluate (BYOE). Provide baseline (`--baseline`) and
subject (`--subject`) checkpoints and InvarLock will run paired windows and emit a
evaluation report. Optionally, pass `--edit-config` to run the built‑in quant_rtn demo.

Steps:
  1) Baseline (no-op edit) on baseline model
  2) Subject (no-op or provided edit config) on subject model with --baseline pairing
  3) Emit evaluation report via `invarlock report --format report`
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import typer
import yaml
from rich.console import Console

from invarlock import __version__ as INVARLOCK_VERSION
from invarlock.clean_pruning_selection_runtime import (
    finalize_clean_pruning_selection_evaluation_report,
    load_clean_pruning_selection_evaluation_context,
)
from invarlock.clean_selection_runtime import (
    finalize_clean_selection_evaluation_report,
    load_clean_selection_evaluation_context,
)
from invarlock.cli import output as cli_output
from invarlock.core.exceptions import resolve_command_exit_code
from invarlock.evidence_catalog_binding import evaluation_input_binding_errors
from invarlock.evidence_pack_json import StrictJsonError, load_json_object
from invarlock.runtime_security import (
    RuntimeManifestExecution,
    current_runtime_security_policy,
)
from invarlock.strict_yaml import StrictYamlError, load_yaml_object

from ...adapters.auto import resolve_auto_adapter
from ...core.evaluate_contract import (
    apply_edited_primary_metric_policy,
)
from ...core.evaluate_plan import (
    build_evaluate_command_plan,
    normalize_model_id,
    resolve_evaluate_execution_policy,
    resolve_evaluate_tmp_dir,
)
from ...core.exceptions import ConfigError, ValidationError

# Use the report group's programmatic entry for report generation
from ...reporting.report_contract import generate_reports
from ..evaluate_output import (
    _evaluation_report_manifest_execution as _evaluation_report_manifest_execution_impl,
)
from ..evaluate_output import _format_ratio as _format_ratio_impl
from ..evaluate_output import _override_console as _override_console_impl
from ..evaluate_output import _phase_title as _phase_title_impl
from ..evaluate_output import _print_header_banner as _print_header_banner_impl
from ..evaluate_output import _print_phase_header as _print_phase_header_impl
from ..evaluate_output import _print_quiet_summary as _print_quiet_summary_impl
from ..evaluate_output import _render_banner_lines as _render_banner_lines_impl
from ..evaluate_output import _resolve_verbosity as _resolve_verbosity_impl
from ..evaluate_output import _suppress_child_output as _suppress_child_output_impl
from ..evaluate_phases import (
    BaselineEvaluationRequest,
    EvaluatePhaseRuntime,
    SubjectEvaluationRequest,
    run_baseline_evaluation_phase,
    run_subject_evaluation_phase,
)
from ..evaluate_report_phase import (
    EvaluationReportRequest,
    EvaluationReportRuntime,
    emit_evaluation_report_phase,
)
from ..evaluate_selection_phase import (
    EvaluateSelectionRequest,
    EvaluateSelectionRuntime,
    SelectionArtifactInputs,
    load_evaluate_selection_contexts,
)
from ..security_helpers import (
    emit_runtime_manifest,
    maybe_delegate_model_command,
    runtime_security_scoped,
)

_LAZY_RUN_IMPORT = True

PHASE_BAR_WIDTH = 67
VERBOSITY_QUIET = 0
VERBOSITY_DEFAULT = 1
VERBOSITY_VERBOSE = 2
_QUIET_REPORT_LOAD_ERRORS = (json.JSONDecodeError, OSError, TypeError, ValueError)
_CONSOLE_SUMMARY_ERRORS = (
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)
_CHILD_RUN_REPLAY_ERRORS = (
    ConfigError,
    OSError,
    RuntimeError,
    TypeError,
    ValidationError,
    ValueError,
)
_EDIT_CONFIG_LOAD_ERRORS = (OSError, TypeError, ValueError, yaml.YAMLError)
_TEXT_NORMALIZATION_ERRORS = (RuntimeError, TypeError, ValueError)
_RUN_TIMING_KEYS = (
    "load_model",
    "load_dataset",
    "prepare",
    "prepare_guards",
    "edit",
    "guards",
    "eval",
    "finalize",
    "execute",
    "total",
)

console = Console()


def _normalize_model_id(model_id: str, adapter: str) -> str:
    return normalize_model_id(model_id, adapter)


def _resolve_evaluate_tmp_dir() -> Path:
    return resolve_evaluate_tmp_dir(os.environ.get("INVARLOCK_EVALUATE_TMP_DIR"))


def _resolve_evaluate_runtime_provider_selection(
    plan: Any,
    *,
    debug_fn: Any,
) -> str:
    debug_fn(
        "Runtime providers -> "
        f"baseline={plan.baseline_runtime_provider_name}, "
        f"subject={plan.subject_runtime_provider_name}"
    )
    return str(plan.subject_runtime_provider_name)


def _require_supported_evaluate_runtime_providers(plan: Any, *, fail_fn: Any) -> None:
    """Fail before execution when the legacy path cannot honor a selection."""

    selected = (
        ("baseline", str(plan.baseline_runtime_provider_name)),
        ("subject", str(plan.subject_runtime_provider_name)),
    )
    unsupported = tuple(
        f"{side}={provider}"
        for side, provider in selected
        if provider != "hf_transformers"
    )
    if unsupported:
        fail_fn(
            "The evaluate execution path currently supports only the "
            "'hf_transformers' runtime provider; unsupported selection(s): "
            + ", ".join(unsupported),
            exit_code=2,
        )


def _render_banner_lines(title: str, context: str) -> list[str]:
    return _render_banner_lines_impl(title, context)


def _print_header_banner(
    console: Console, *, version: str, profile: str, tier: str, adapter: str
) -> None:
    _print_header_banner_impl(
        console, version=version, profile=profile, tier=tier, adapter=adapter
    )


def _phase_title(index: int, total: int, title: str) -> str:
    return _phase_title_impl(index, total, title)


def _print_phase_header(console: Console, title: str) -> None:
    _print_phase_header_impl(console, title)


def _format_ratio(value: Any) -> str:
    return _format_ratio_impl(value)


def _coerce_timing_seconds(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_report_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = load_json_object(path, label="run report")
    except StrictJsonError:
        return None
    return payload


def _extract_run_timings_seconds(payload: dict[str, Any] | None) -> dict[str, float]:
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    timings = metrics.get("timings") if isinstance(metrics, dict) else None
    if not isinstance(timings, dict):
        return {}
    extracted: dict[str, float] = {}
    for key in _RUN_TIMING_KEYS:
        value = _coerce_timing_seconds(timings.get(key))
        if value is not None:
            extracted[key] = max(0.0, value)
    return extracted


def _aggregate_run_timings_seconds(
    run_timings: dict[str, dict[str, float]],
) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    for timings in run_timings.values():
        for key, value in timings.items():
            aggregate[key] = aggregate.get(key, 0.0) + float(value)
    return aggregate


def _evaluation_report_manifest_execution(
    *,
    execution_mode: str,
    allow_network: bool,
    allow_remote_code: bool,
    allow_third_party_plugins: bool,
) -> RuntimeManifestExecution | None:
    return _evaluation_report_manifest_execution_impl(
        execution_mode=execution_mode,
        allow_network=allow_network,
        allow_remote_code=allow_remote_code,
        allow_third_party_plugins=allow_third_party_plugins,
    )


def _resolve_verbosity(quiet: bool, verbose: bool) -> int:
    return _resolve_verbosity_impl(quiet, verbose, console=console)


@contextmanager
def _override_console(module: Any, new_console: Console) -> Iterator[None]:
    with _override_console_impl(module, new_console):
        yield


@contextmanager
def _suppress_child_output(enabled: bool) -> Iterator[io.StringIO | None]:
    from .. import run_execution as run_exec_mod
    from . import report as report_mod
    from . import run as run_mod

    with _suppress_child_output_impl(
        enabled,
        run_execution_module=run_exec_mod,
        report_module=report_mod,
        run_module=run_mod,
    ) as buffer:
        yield buffer


def _print_quiet_summary(
    *,
    report_out: Path,
    baseline: str,
    subject: str,
    profile: str,
) -> None:
    _print_quiet_summary_impl(
        console=console,
        report_out=report_out,
        baseline=baseline,
        subject=subject,
        profile=profile,
    )


def _release_phase_memory() -> None:
    from .. import run_runtime_exec as run_runtime_mod

    run_runtime_mod.release_process_memory()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        return load_yaml_object(path, label="Preset")
    except StrictYamlError as exc:
        raise ValueError(str(exc)) from exc


def _load_json_object_path(path: Path) -> dict[str, Any]:
    return load_json_object(path, label="edited run report")


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _build_evaluate_phase_runtime(
    *,
    console: Console,
    output_style: Any,
    timings: dict[str, float],
    verbosity: int,
    progress: bool,
    info_fn: Any,
    debug_fn: Any,
    phase_fn: Any,
    fail_fn: Any,
) -> EvaluatePhaseRuntime:
    from . import run as run_mod

    return EvaluatePhaseRuntime(
        console=console,
        output_style=output_style,
        timings=timings,
        verbosity=verbosity,
        progress=progress,
        info_fn=info_fn,
        debug_fn=debug_fn,
        phase_fn=phase_fn,
        fail_fn=fail_fn,
        suppress_child_output_fn=_suppress_child_output,
        load_yaml_fn=_load_yaml,
        dump_yaml_fn=_dump_yaml,
        run_command_fn=run_mod.run_command,
        json_load_fn=_load_json_object_path,
    )


def _selection_contexts(
    *,
    baseline: str,
    subject: str,
    assurance: str,
    allow_network: bool,
    allow_remote_code: bool,
    allow_third_party_plugins: bool,
    execution_policy: Any,
    clean_selection: SelectionArtifactInputs,
    clean_pruning_selection: SelectionArtifactInputs,
) -> tuple[Any, Any]:
    """Load optional transformation-selection evidence at the CLI boundary."""

    return load_evaluate_selection_contexts(
        EvaluateSelectionRequest(
            baseline=baseline,
            subject=subject,
            assurance=assurance,
            allow_network=allow_network,
            allow_remote_code=allow_remote_code,
            allow_third_party_plugins=allow_third_party_plugins,
            clean_selection=clean_selection,
            clean_pruning_selection=clean_pruning_selection,
        ),
        EvaluateSelectionRuntime(
            execution_policy=execution_policy,
            current_security_policy_fn=current_runtime_security_policy,
            delegate_model_command_fn=maybe_delegate_model_command,
            load_clean_selection_fn=load_clean_selection_evaluation_context,
            load_clean_pruning_selection_fn=(
                load_clean_pruning_selection_evaluation_context
            ),
        ),
    )


@runtime_security_scoped
def evaluate_command(
    baseline: str,
    subject: str,
    baseline_report: str | None = None,
    baseline_adapter: str = "auto",
    subject_adapter: str = "auto",
    baseline_runtime_provider: str = "hf_transformers",
    subject_runtime_provider: str = "hf_transformers",
    device: str | None = None,
    profile: str = "ci",
    tier: str = "balanced",
    preset: str | None = None,
    evaluation_input_binding: str | None = None,
    out: str = "runs",
    report_out: str = "reports/eval",
    edit_config: str | None = None,
    edit_label: str | None = None,
    quiet: bool = False,
    verbose: bool = False,
    banner: bool = True,
    style: str = "audit",
    timing: bool = False,
    timing_json: str | None = None,
    progress: bool = True,
    execution_mode: str = "container",
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    assurance: str = "strict",
    defer_report_rendering: bool = False,
    no_color: bool = False,
    baseline_revision: str | None = None,
    subject_revision: str | None = None,
    clean_selection_config: str | None = None,
    clean_selection_execution_receipt: str | None = None,
    clean_selection_replay: str | None = None,
    clean_selection_runtime_proof: str | None = None,
    clean_selection_repeat_index: int | None = None,
    clean_pruning_selection_config: str | None = None,
    clean_pruning_selection_execution_receipt: str | None = None,
    clean_pruning_selection_replay: str | None = None,
    clean_pruning_selection_runtime_proof: str | None = None,
    clean_pruning_selection_repeat_index: int | None = None,
):
    """Evaluate two checkpoints (baseline vs subject) with pinned windows."""
    try:
        execution_policy = resolve_evaluate_execution_policy(
            execution_mode=execution_mode,
            allow_host_execution=allow_host_execution,
        )
    except ValueError as exc:
        raise typer.BadParameter(
            "Execution mode must be one of: container, host.",
            param_hint="--execution-mode",
        ) from exc
    allow_host_execution = execution_policy.allow_host_execution
    prefer_local_files_only = execution_policy.prefer_local_files_only
    allow_unverified_provenance = execution_policy.allow_unverified_provenance

    clean_selection_context, clean_pruning_selection_context = _selection_contexts(
        baseline=baseline,
        subject=subject,
        assurance=assurance,
        allow_network=allow_network,
        allow_remote_code=allow_remote_code,
        allow_third_party_plugins=allow_third_party_plugins,
        execution_policy=execution_policy,
        clean_selection=SelectionArtifactInputs(
            config=clean_selection_config,
            execution_receipt=clean_selection_execution_receipt,
            replay=clean_selection_replay,
            runtime_proof=clean_selection_runtime_proof,
            repeat_index=clean_selection_repeat_index,
        ),
        clean_pruning_selection=SelectionArtifactInputs(
            config=clean_pruning_selection_config,
            execution_receipt=clean_pruning_selection_execution_receipt,
            replay=clean_pruning_selection_replay,
            runtime_proof=clean_pruning_selection_runtime_proof,
            repeat_index=clean_pruning_selection_repeat_index,
        ),
    )

    verbosity = _resolve_verbosity(bool(quiet), bool(verbose))

    if verbosity == VERBOSITY_QUIET:
        progress = False
        timing = False

    output_style = cli_output.resolve_output_style(
        style=str(style),
        profile=str(profile),
        progress=bool(progress),
        timing=bool(timing),
        no_color=bool(no_color),
    )
    console = cli_output.make_console(no_color=not output_style.color)
    timings: dict[str, float] = {}
    total_start: float | None = (
        cli_output.perf_counter() if output_style.timing or timing_json else None
    )

    def _info(message: str, *, tag: str = "INFO", emoji: str | None = None) -> None:
        if verbosity >= VERBOSITY_DEFAULT:
            cli_output.print_event(
                console, tag, message, style=output_style, emoji=emoji
            )

    def _debug(msg: str) -> None:
        if verbosity >= VERBOSITY_VERBOSE:
            console.print(msg, markup=False)

    def _fail(message: str, *, exit_code: int = 2) -> NoReturn:
        cli_output.print_event(console, "FAIL", message, style=output_style, emoji="❌")
        raise typer.Exit(exit_code)

    def _phase(index: int, total: int, title: str) -> None:
        if verbosity >= VERBOSITY_DEFAULT:
            console.print("")
            _print_phase_header(console, _phase_title(index, total, title))

    src_id = str(baseline)
    edt_id = str(subject)
    evaluation_binding_payload: dict[str, object] | None = None
    if evaluation_input_binding is not None:
        try:
            loaded_binding = _load_json_object_path(Path(evaluation_input_binding))
        except (OSError, StrictJsonError, TypeError, ValueError) as exc:
            _fail(f"Evaluation input binding cannot be loaded: {exc}", exit_code=2)
        binding_errors = evaluation_input_binding_errors(loaded_binding)
        if binding_errors:
            _fail("; ".join(binding_errors), exit_code=2)
        evaluation_binding_payload = loaded_binding
    plan_start: float | None = (
        cli_output.perf_counter() if total_start is not None else None
    )
    try:
        plan = build_evaluate_command_plan(
            baseline_model_id=src_id,
            subject_model_id=edt_id,
            baseline_revision=baseline_revision,
            subject_revision=subject_revision,
            baseline_adapter=baseline_adapter,
            subject_adapter=subject_adapter,
            baseline_runtime_provider=baseline_runtime_provider,
            subject_runtime_provider=subject_runtime_provider,
            profile=profile,
            tier=tier,
            preset=preset,
            out=out,
            edit_config=edit_config,
            edit_label=edit_label,
            resolve_auto_adapter_fn=resolve_auto_adapter,
            load_yaml_fn=_load_yaml,
            tmp_dir_candidate=os.environ.get("INVARLOCK_EVALUATE_TMP_DIR"),
            assurance_mode=assurance,
            execution_mode=execution_mode,
            allow_unverified_provenance=allow_unverified_provenance,
            evaluation_input_binding=evaluation_binding_payload,
        )
    except FileNotFoundError as exc:
        _fail(f"Preset not found: {exc}", exit_code=2)
    except ValueError as exc:
        _fail(str(exc), exit_code=2)
    _require_supported_evaluate_runtime_providers(plan, fail_fn=_fail)
    if plan_start is not None:
        timings["plan"] = max(0.0, float(cli_output.perf_counter() - plan_start))
    profile_name = plan.profile_name
    tier_name = plan.tier_name
    baseline_eff_adapter = plan.baseline_adapter_name
    subject_eff_adapter = plan.subject_adapter_name
    subject_eff_runtime_provider = _resolve_evaluate_runtime_provider_selection(
        plan, debug_fn=_debug
    )
    adapter_auto = plan.adapter_auto
    adapter_display = (
        subject_eff_adapter
        if baseline_eff_adapter == subject_eff_adapter
        else f"{baseline_eff_adapter}->{subject_eff_adapter}"
    )

    show_banner = bool(banner) and verbosity >= VERBOSITY_DEFAULT
    if show_banner:
        _print_header_banner(
            console,
            version=INVARLOCK_VERSION,
            profile=profile_name,
            tier=tier_name,
            adapter=str(adapter_display),
        )
        console.print("")

    if adapter_auto:
        _debug(
            "Adapter:auto -> "
            f"baseline={baseline_eff_adapter}, subject={subject_eff_adapter}"
        )
    # Choose preset. If none provided and repo preset is missing (pip install
    # scenario), fall back to a minimal built-in universal preset so the
    # flag-only quick start works without cloning the repo.
    preset_data = plan.preset_data
    guards_order = plan.guards_order
    norm_edt_id = plan.subject_model_id
    baseline_cfg = plan.baseline_config
    baseline_label = plan.baseline_label
    subject_label = plan.subject_label
    assurance_mode = plan.assurance_mode
    tmp_dir = plan.tmp_dir

    phase_runtime = _build_evaluate_phase_runtime(
        console=console,
        output_style=output_style,
        timings=timings,
        verbosity=verbosity,
        progress=progress,
        info_fn=_info,
        debug_fn=_debug,
        phase_fn=_phase,
        fail_fn=_fail,
    )

    baseline_report_path = run_baseline_evaluation_phase(
        BaselineEvaluationRequest(
            baseline_report=baseline_report,
            profile_name=profile_name,
            tier_name=tier_name,
            adapter=str(baseline_eff_adapter),
            out=out,
            device=device,
            allow_network=allow_network,
            allow_host_execution=allow_host_execution,
            allow_third_party_plugins=allow_third_party_plugins,
            allow_remote_code=allow_remote_code,
            allow_unverified_provenance=allow_unverified_provenance,
            prefer_local_files_only=prefer_local_files_only,
            no_color=no_color,
            baseline_cfg=baseline_cfg,
            baseline_label=baseline_label,
            tmp_dir=tmp_dir,
        ),
        phase_runtime,
    )
    _release_phase_memory()

    edited_report, edited_payload, resolved_subject_config = (
        run_subject_evaluation_phase(
            SubjectEvaluationRequest(
                baseline_report_path=baseline_report_path,
                preset_data=preset_data,
                subject_model_id=norm_edt_id,
                adapter=str(subject_eff_adapter),
                out=out,
                device=device,
                profile_name=profile_name,
                tier_name=tier_name,
                guards_order=guards_order,
                assurance_mode=assurance_mode,
                subject_label=subject_label,
                edit_config=edit_config,
                edit_label=edit_label,
                execution_mode=execution_mode,
                allow_network=allow_network,
                allow_host_execution=allow_host_execution,
                allow_third_party_plugins=allow_third_party_plugins,
                allow_remote_code=allow_remote_code,
                allow_unverified_provenance=allow_unverified_provenance,
                prefer_local_files_only=prefer_local_files_only,
                no_color=no_color,
                tmp_dir=tmp_dir,
                model_identity=plan.subject_identity,
                runtime_provider=subject_eff_runtime_provider,
            ),
            phase_runtime,
        )
    )
    _release_phase_memory()

    _phase(3, 3, "EVALUATION REPORT GENERATION")

    # CI/Release hard‑abort: fail fast when primary metric is not computable.
    try:
        prof = str(profile or "").strip().lower()
    except _TEXT_NORMALIZATION_ERRORS:
        prof = ""
    if prof in {"ci", "ci_cpu", "release"}:
        outcome = apply_edited_primary_metric_policy(
            edited_payload,
            profile=profile,
        )
        if outcome.diagnostic is not None and outcome.error is not None:
            cli_output.print_event(
                console,
                "WARN",
                outcome.diagnostic.message,
                style=output_style,
                emoji="⚠️",
            )
            raise typer.Exit(resolve_command_exit_code(outcome.error, profile=profile))

    emit_evaluation_report_phase(
        EvaluationReportRequest(
            edited_report=edited_report,
            resolved_subject_config=resolved_subject_config,
            baseline_report_path=baseline_report_path,
            report_out=report_out,
            baseline=baseline,
            subject=subject,
            baseline_eff_adapter=str(baseline_eff_adapter),
            subject_eff_adapter=str(subject_eff_adapter),
            profile_name=profile_name,
            tier_name=tier_name,
            preset=preset,
            out=out,
            edit_config=edit_config,
            edit_label=edit_label,
            allow_network=allow_network,
            allow_remote_code=allow_remote_code,
            allow_third_party_plugins=allow_third_party_plugins,
            execution_mode=execution_mode,
            assurance_mode=assurance_mode,
            defer_report_rendering=defer_report_rendering,
            clean_selection_context=clean_selection_context,
            clean_pruning_selection_context=clean_pruning_selection_context,
        ),
        EvaluationReportRuntime(
            console=console,
            output_style=output_style,
            timings=timings,
            info_fn=_info,
            fail_fn=_fail,
            generate_reports_fn=generate_reports,
            emit_runtime_manifest_fn=emit_runtime_manifest,
            manifest_execution_fn=_evaluation_report_manifest_execution,
            finalize_clean_selection_report_fn=finalize_clean_selection_evaluation_report,
            finalize_clean_pruning_selection_report_fn=(
                finalize_clean_pruning_selection_evaluation_report
            ),
        ),
    )
    if total_start is not None:
        timings["total"] = max(0.0, float(cli_output.perf_counter() - total_start))
    else:
        timings["total"] = (
            float(timings.get("baseline", 0.0))
            + float(timings.get("subject", 0.0))
            + float(timings.get("evaluation_report", 0.0))
        )
    if timing:
        cli_output.print_timing_summary(
            console,
            timings,
            style=output_style,
            order=[
                ("Plan", "plan"),
                ("Baseline", "baseline"),
                ("Subject", "subject"),
                ("Evaluation Report", "evaluation_report"),
                ("Total", "total"),
            ],
        )
    if timing_json:
        baseline_payload = _load_report_payload(baseline_report_path)
        run_timings_seconds = {
            "baseline": _extract_run_timings_seconds(baseline_payload),
            "subject": _extract_run_timings_seconds(edited_payload),
        }
        aggregate_run_timings = _aggregate_run_timings_seconds(run_timings_seconds)
        timing_payload = {
            "schema": "invarlock/evaluate-timing-v1",
            "baseline": baseline,
            "subject": subject,
            "baseline_adapter": str(baseline_eff_adapter),
            "subject_adapter": str(subject_eff_adapter),
            "profile": profile_name,
            "tier": tier_name,
            "baseline_report_reused": bool(baseline_report),
            "defer_report_rendering": bool(defer_report_rendering),
            "timings_seconds": {key: float(value) for key, value in timings.items()},
        }
        if any(run_timings_seconds.values()):
            timing_payload["run_timings_seconds"] = run_timings_seconds
        if aggregate_run_timings:
            timing_payload["aggregate_run_timings_seconds"] = aggregate_run_timings
        timing_path = Path(timing_json)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(
            json.dumps(timing_payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    if verbosity == VERBOSITY_QUIET:
        _print_quiet_summary(
            report_out=Path(report_out),
            baseline=src_id,
            subject=edt_id,
            profile=profile_name,
        )
