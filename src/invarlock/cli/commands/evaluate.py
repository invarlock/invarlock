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
from invarlock.cli import output as cli_output
from invarlock.core.exceptions import resolve_command_exit_code
from invarlock.runtime_security import (
    RuntimeManifestExecution,
)

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
    _run_baseline_evaluation_phase as _run_baseline_evaluation_phase_impl,
)
from ..evaluate_phases import (
    _run_subject_evaluation_phase as _run_subject_evaluation_phase_impl,
)
from ..evaluate_report_phase import (
    emit_evaluation_report_phase as _emit_evaluation_report_phase_impl,
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
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


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
        json_load_fn=json.load,
    )


def _release_phase_memory() -> None:
    from .. import run_runtime_exec as run_runtime_mod

    run_runtime_mod.release_process_memory()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Preset must be a mapping")
    return data


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _run_baseline_evaluation_phase(
    *,
    baseline_report: str | None,
    profile_name: str,
    tier_name: str,
    eff_adapter: str,
    out: str,
    device: str | None,
    allow_network: bool,
    allow_host_execution: bool,
    allow_third_party_plugins: bool,
    allow_remote_code: bool,
    allow_unverified_provenance: bool,
    prefer_local_files_only: bool,
    no_color: bool,
    baseline_cfg: dict[str, Any],
    baseline_label: str,
    tmp_dir: Path,
    console: Console,
    output_style: Any,
    timings: dict[str, float],
    verbosity: int,
    progress: bool,
    info_fn: Any,
    debug_fn: Any,
    phase_fn: Any,
    fail_fn: Any,
) -> Path:
    from . import run as run_mod

    return _run_baseline_evaluation_phase_impl(
        baseline_report=baseline_report,
        profile_name=profile_name,
        tier_name=tier_name,
        eff_adapter=eff_adapter,
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
        dump_yaml_fn=_dump_yaml,
        run_command_fn=run_mod.run_command,
    )


def _run_subject_evaluation_phase(
    *,
    baseline_report_path: Path,
    preset_data: dict[str, Any],
    norm_edt_id: str,
    eff_adapter: str,
    out: str,
    device: str | None,
    profile_name: str,
    tier_name: str,
    guards_order: Any,
    assurance_mode: str,
    subject_label: str | None,
    edit_config: str | None,
    edit_label: str | None,
    console: Console,
    output_style: Any,
    timings: dict[str, float],
    verbosity: int,
    progress: bool,
    execution_mode: str,
    allow_network: bool,
    allow_host_execution: bool,
    allow_third_party_plugins: bool,
    allow_remote_code: bool,
    allow_unverified_provenance: bool,
    prefer_local_files_only: bool,
    no_color: bool,
    tmp_dir: Path,
    info_fn: Any,
    debug_fn: Any,
    phase_fn: Any,
    fail_fn: Any,
) -> tuple[Path, dict[str, Any]]:
    from . import run as run_mod

    return _run_subject_evaluation_phase_impl(
        baseline_report_path=baseline_report_path,
        preset_data=preset_data,
        norm_edt_id=norm_edt_id,
        eff_adapter=eff_adapter,
        out=out,
        device=device,
        profile_name=profile_name,
        tier_name=tier_name,
        guards_order=guards_order,
        assurance_mode=assurance_mode,
        subject_label=subject_label,
        edit_config=edit_config,
        edit_label=edit_label,
        console=console,
        output_style=output_style,
        timings=timings,
        verbosity=verbosity,
        progress=progress,
        execution_mode=execution_mode,
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        allow_unverified_provenance=allow_unverified_provenance,
        prefer_local_files_only=prefer_local_files_only,
        no_color=no_color,
        tmp_dir=tmp_dir,
        info_fn=info_fn,
        debug_fn=debug_fn,
        phase_fn=phase_fn,
        fail_fn=fail_fn,
        load_yaml_fn=_load_yaml,
        dump_yaml_fn=_dump_yaml,
        suppress_child_output_fn=_suppress_child_output,
        run_command_fn=run_mod.run_command,
        json_load_fn=json.load,
    )


@runtime_security_scoped
def evaluate_command(
    baseline: str,
    subject: str,
    baseline_report: str | None = None,
    baseline_adapter: str = "auto",
    subject_adapter: str = "auto",
    device: str | None = None,
    profile: str = "ci",
    tier: str = "balanced",
    preset: str | None = None,
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
    maybe_delegate_model_command()

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
    plan_start: float | None = (
        cli_output.perf_counter() if total_start is not None else None
    )
    try:
        plan = build_evaluate_command_plan(
            baseline_model_id=src_id,
            subject_model_id=edt_id,
            baseline_adapter=baseline_adapter,
            subject_adapter=subject_adapter,
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
        )
    except FileNotFoundError as exc:
        _fail(f"Preset not found: {exc}", exit_code=2)
    except ValueError as exc:
        _fail(str(exc), exit_code=2)
    if plan_start is not None:
        timings["plan"] = max(0.0, float(cli_output.perf_counter() - plan_start))
    profile_name = plan.profile_name
    tier_name = plan.tier_name
    baseline_eff_adapter = plan.baseline_adapter_name
    subject_eff_adapter = plan.subject_adapter_name
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

    baseline_report_path = _run_baseline_evaluation_phase(
        baseline_report=baseline_report,
        profile_name=profile_name,
        tier_name=tier_name,
        eff_adapter=str(baseline_eff_adapter),
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
    _release_phase_memory()

    edited_report, edited_payload = _run_subject_evaluation_phase(
        baseline_report_path=baseline_report_path,
        preset_data=preset_data,
        norm_edt_id=norm_edt_id,
        eff_adapter=str(subject_eff_adapter),
        out=out,
        device=device,
        profile_name=profile_name,
        tier_name=tier_name,
        guards_order=guards_order,
        assurance_mode=assurance_mode,
        subject_label=subject_label,
        edit_config=edit_config,
        edit_label=edit_label,
        console=console,
        output_style=output_style,
        timings=timings,
        verbosity=verbosity,
        progress=progress,
        execution_mode=execution_mode,
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        allow_unverified_provenance=allow_unverified_provenance,
        prefer_local_files_only=prefer_local_files_only,
        no_color=no_color,
        tmp_dir=tmp_dir,
        info_fn=_info,
        debug_fn=_debug,
        phase_fn=_phase,
        fail_fn=_fail,
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

    _emit_evaluation_report_phase_impl(
        edited_report=edited_report,
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
        console=console,
        output_style=output_style,
        timings=timings,
        info_fn=_info,
        fail_fn=_fail,
        generate_reports_fn=generate_reports,
        emit_runtime_manifest_fn=emit_runtime_manifest,
        manifest_execution_fn=_evaluation_report_manifest_execution,
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
            json.dumps(timing_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if verbosity == VERBOSITY_QUIET:
        _print_quiet_summary(
            report_out=Path(report_out),
            baseline=src_id,
            subject=edt_id,
            profile=profile_name,
        )
