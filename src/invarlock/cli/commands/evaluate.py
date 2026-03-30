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
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import typer
import yaml
from rich.console import Console

from invarlock import __version__ as INVARLOCK_VERSION
from invarlock.exit_codes import resolve_command_exit_code

from ...core.adapter_auto import resolve_auto_adapter
from ...core.evaluate_contract import (
    apply_edited_primary_metric_policy,
    load_validated_baseline_report,
    require_run_report_artifact,
)
from ...core.evaluate_plan import (
    build_evaluate_command_plan,
    build_subject_edit_run_config,
    build_subject_noop_run_config,
    normalize_model_id,
    resolve_evaluate_execution_policy,
    resolve_evaluate_tmp_dir,
)
from ...core.exceptions import ConfigError, ValidationError

# Use the report group's programmatic entry for report generation
from ...reporting.report_contract import generate_reports
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

console = Console()


def _normalize_model_id(model_id: str, adapter: str) -> str:
    return normalize_model_id(model_id, adapter)


def _resolve_evaluate_tmp_dir() -> Path:
    return resolve_evaluate_tmp_dir(os.environ.get("INVARLOCK_EVALUATE_TMP_DIR"))


def _render_banner_lines(title: str, context: str) -> list[str]:
    return [
        title,
        context,
        "-" * max(len(title), len(context)),
    ]


def _print_header_banner(
    console: Console, *, version: str, profile: str, tier: str, adapter: str
) -> None:
    title = f"INVARLOCK v{version} · Evaluation Pipeline"
    context = f"Profile: {profile} · Tier: {tier} · Adapter: {adapter}"
    for line in _render_banner_lines(title, context):
        console.print(line)


def _phase_title(index: int, total: int, title: str) -> str:
    return f"PHASE {index}/{total} · {title}"


def _print_phase_header(console: Console, title: str) -> None:
    console.print(title)
    console.print("-" * max(PHASE_BAR_WIDTH, len(title)))


def _format_ratio(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(val):
        return "N/A"
    return f"{val:.3f}"


def _resolve_verbosity(quiet: bool, verbose: bool) -> int:
    if quiet and verbose:
        console.print("--quiet and --verbose are mutually exclusive")
        raise typer.Exit(2)
    if quiet:
        return VERBOSITY_QUIET
    if verbose:
        return VERBOSITY_VERBOSE
    return VERBOSITY_DEFAULT


@contextmanager
def _override_console(module: Any, new_console: Console) -> Iterator[None]:
    original_console = getattr(module, "console", None)
    module.console = new_console
    try:
        yield
    finally:
        module.console = original_console


@contextmanager
def _suppress_child_output(enabled: bool) -> Iterator[io.StringIO | None]:
    if not enabled:
        yield None
        return
    from .. import run_execution as run_exec_mod
    from . import report as report_mod
    from . import run as run_mod

    buffer = io.StringIO()
    quiet_console = Console(file=buffer, force_terminal=False, color_system=None)
    with (
        _override_console(run_mod, quiet_console),
        _override_console(run_exec_mod, quiet_console),
        _override_console(report_mod, quiet_console),
    ):
        yield buffer


def _print_quiet_summary(
    *,
    report_out: Path,
    baseline: str,
    subject: str,
    profile: str,
) -> None:
    report_path = report_out / "evaluation.report.json"
    console.print(f"INVARLOCK v{INVARLOCK_VERSION} · EVALUATE")
    console.print(f"Baseline: {baseline} -> Subject: {subject} · Profile: {profile}")
    if not report_path.exists():
        console.print(f"Output: {report_out}")
        return
    try:
        with report_path.open("r", encoding="utf-8") as fh:
            evaluation_report = json.load(fh)
    except _QUIET_REPORT_LOAD_ERRORS:
        console.print(f"Output: {report_path}")
        return
    if not isinstance(evaluation_report, dict):
        console.print(f"Output: {report_path}")
        return
    try:
        from invarlock.reporting.report_console import (
            compute_console_validation_block as _console_block,
        )

        block = _console_block(evaluation_report)
        rows = block.get("rows", [])
        total = len(rows) if isinstance(rows, list) else 0
        passed = (
            sum(1 for row in rows if row.get("ok")) if isinstance(rows, list) else 0
        )
        status = "PASS" if block.get("overall_pass") else "FAIL"
    except _CONSOLE_SUMMARY_ERRORS:
        total = 0
        passed = 0
        status = "UNKNOWN"
    pm_ratio = _format_ratio(
        (evaluation_report.get("primary_metric") or {}).get("ratio_vs_baseline")
    )
    gate_summary = f"{passed}/{total} passed" if total else "N/A"
    console.print(f"Status: {status} · Gates: {gate_summary}")
    if pm_ratio != "N/A":
        console.print(f"Primary metric ratio: {pm_ratio}")
    console.print(f"Output: {report_path}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Preset must be a mapping")
    return data


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


@runtime_security_scoped
def evaluate_command(
    baseline: str,
    subject: str,
    baseline_report: str | None = None,
    adapter: str = "auto",
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
    progress: bool = True,
    mode: str = "attested",
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    no_color: bool = False,
):
    """Evaluate two checkpoints (baseline vs subject) with pinned windows."""
    try:
        execution_policy = resolve_evaluate_execution_policy(
            mode=mode,
            allow_host_execution=allow_host_execution,
        )
    except ValueError as exc:
        raise typer.BadParameter(
            "Execution mode must be one of: attested, local.",
            param_hint="--mode",
        ) from exc
    allow_host_execution = execution_policy.allow_host_execution
    prefer_local_files_only = execution_policy.prefer_local_files_only
    maybe_delegate_model_command()

    verbosity = _resolve_verbosity(bool(quiet), bool(verbose))

    if verbosity == VERBOSITY_QUIET:
        progress = False
        timing = False

    from invarlock.cli.output import (
        make_console,
        perf_counter,
        print_event,
        print_timing_summary,
        resolve_output_style,
        timed_step,
    )

    output_style = resolve_output_style(
        style=str(style),
        profile=str(profile),
        progress=bool(progress),
        timing=bool(timing),
        no_color=bool(no_color),
    )
    console = make_console(no_color=not output_style.color)
    timings: dict[str, float] = {}
    total_start: float | None = perf_counter() if output_style.timing else None

    def _info(message: str, *, tag: str = "INFO", emoji: str | None = None) -> None:
        if verbosity >= VERBOSITY_DEFAULT:
            print_event(console, tag, message, style=output_style, emoji=emoji)

    def _debug(msg: str) -> None:
        if verbosity >= VERBOSITY_VERBOSE:
            console.print(msg, markup=False)

    def _fail(message: str, *, exit_code: int = 2) -> NoReturn:
        print_event(console, "FAIL", message, style=output_style, emoji="❌")
        raise typer.Exit(exit_code)

    def _phase(index: int, total: int, title: str) -> None:
        if verbosity >= VERBOSITY_DEFAULT:
            console.print("")
            _print_phase_header(console, _phase_title(index, total, title))

    src_id = str(baseline)
    edt_id = str(subject)
    try:
        plan = build_evaluate_command_plan(
            baseline_model_id=src_id,
            subject_model_id=edt_id,
            adapter=adapter,
            profile=profile,
            tier=tier,
            preset=preset,
            out=out,
            edit_config=edit_config,
            edit_label=edit_label,
            resolve_auto_adapter_fn=resolve_auto_adapter,
            load_yaml_fn=_load_yaml,
            tmp_dir_candidate=os.environ.get("INVARLOCK_EVALUATE_TMP_DIR"),
        )
    except FileNotFoundError as exc:
        _fail(f"Preset not found: {exc}", exit_code=2)
    profile_name = plan.profile_name
    tier_name = plan.tier_name
    eff_adapter = plan.adapter_name
    adapter_auto = plan.adapter_auto

    show_banner = bool(banner) and verbosity >= VERBOSITY_DEFAULT
    if show_banner:
        _print_header_banner(
            console,
            version=INVARLOCK_VERSION,
            profile=profile_name,
            tier=tier_name,
            adapter=str(eff_adapter),
        )
        console.print("")

    if adapter_auto:
        _debug(f"Adapter:auto -> {eff_adapter}")

    # Choose preset. If none provided and repo preset is missing (pip install
    # scenario), fall back to a minimal built-in universal preset so the
    # flag-only quick start works without cloning the repo.
    preset_data = plan.preset_data
    guards_order = plan.guards_order
    norm_edt_id = plan.subject_model_id
    baseline_cfg = plan.baseline_config
    baseline_label = plan.baseline_label
    subject_label = plan.subject_label
    tmp_dir = plan.tmp_dir

    baseline_report_path: Path
    if baseline_report:
        _info(
            "Using provided baseline report (skipping baseline evaluation)",
            tag="EXEC",
            emoji="♻️",
        )
        try:
            baseline_report_path, _ = load_validated_baseline_report(
                Path(baseline_report),
                expected_profile=profile_name,
                expected_tier=tier_name,
                expected_adapter=str(eff_adapter),
            )
        except ValidationError as exc:
            _fail(str(getattr(exc, "message", exc)), exit_code=2)
        _debug(f"Baseline report: {baseline_report_path}")
    else:
        baseline_yaml = tmp_dir / "baseline_noop.yaml"
        _dump_yaml(baseline_yaml, baseline_cfg)

        _phase(1, 3, "BASELINE EVALUATION")
        _info("Running baseline (no-op edit)", tag="EXEC", emoji="🏁")
        _debug(f"Baseline config: {baseline_yaml}")
        from .run import run_command as _run

        with _suppress_child_output(verbosity == VERBOSITY_QUIET) as quiet_buffer:
            try:
                with timed_step(
                    console=console,
                    style=output_style,
                    timings=timings,
                    key="baseline",
                    tag="EXEC",
                    message="Baseline",
                    emoji="🏁",
                ):
                    baseline_run_result = _run(
                        config=str(baseline_yaml),
                        profile=profile_name,
                        out=str(Path(out) / "source"),
                        tier=tier_name,
                        device=device,
                        until_pass=False,
                        max_attempts=1,
                        timeout=None,
                        edit_label=baseline_label,
                        style=output_style.name,
                        progress=progress,
                        timing=False,
                        allow_network=allow_network,
                        allow_host_execution=allow_host_execution,
                        allow_third_party_plugins=allow_third_party_plugins,
                        allow_remote_code=allow_remote_code,
                        prefer_local_files_only=prefer_local_files_only,
                        no_color=no_color,
                    )
            except typer.Exit:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise
            except _CHILD_RUN_REPLAY_ERRORS:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise

        try:
            baseline_report_path = require_run_report_artifact(
                baseline_run_result,
                stage="Baseline",
            )
        except ConfigError as exc:
            _fail(str(getattr(exc, "message", exc)), exit_code=1)
        _debug(f"Baseline report: {baseline_report_path}")

    # Edited run: either no-op (Compare & Evaluate) or provided edit_config (demo edit)
    _phase(2, 3, "SUBJECT EVALUATION")
    if edit_config:
        edited_yaml = Path(edit_config)
        if not edited_yaml.exists():
            print_event(
                console,
                "FAIL",
                f"Edit config not found: {edited_yaml}",
                style=output_style,
                emoji="❌",
            )
            raise typer.Exit(1)
        _info("Running edited (demo edit via --edit-config)", tag="EXEC", emoji="✂️")
        # Overlay subject model id/adapter and output/context onto the provided edit config
        try:
            cfg_loaded: dict[str, Any] = _load_yaml(edited_yaml)
        except _EDIT_CONFIG_LOAD_ERRORS as exc:
            print_event(
                console,
                "FAIL",
                f"Failed to load edit config: {exc}",
                style=output_style,
                emoji="❌",
            )
            raise typer.Exit(1) from exc

        merged_edited_cfg = build_subject_edit_run_config(
            preset_data,
            cfg_loaded,
            subject_model_id=norm_edt_id,
            adapter_name=str(eff_adapter),
            output_dir=str(Path(out) / "edited"),
            profile=profile_name,
            tier=tier_name,
            guards_order=guards_order,
        )

        # Persist a temporary merged config for traceability
        edited_merged_yaml = tmp_dir / "edited_merged.yaml"
        _dump_yaml(edited_merged_yaml, merged_edited_cfg)
        _debug(f"Edited config (merged): {edited_merged_yaml}")

        from .run import run_command as _run

        with _suppress_child_output(verbosity == VERBOSITY_QUIET) as quiet_buffer:
            try:
                with timed_step(
                    console=console,
                    style=output_style,
                    timings=timings,
                    key="subject",
                    tag="EXEC",
                    message="Subject",
                    emoji="✂️",
                ):
                    edited_run_result = _run(
                        config=str(edited_merged_yaml),
                        profile=profile_name,
                        out=str(Path(out) / "edited"),
                        tier=tier_name,
                        baseline=str(baseline_report_path),
                        device=device,
                        until_pass=False,
                        max_attempts=1,
                        timeout=None,
                        edit_label=subject_label if edit_label else None,
                        style=output_style.name,
                        progress=progress,
                        timing=False,
                        allow_network=allow_network,
                        allow_host_execution=allow_host_execution,
                        allow_third_party_plugins=allow_third_party_plugins,
                        allow_remote_code=allow_remote_code,
                        prefer_local_files_only=prefer_local_files_only,
                        no_color=no_color,
                    )
            except typer.Exit:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise
            except _CHILD_RUN_REPLAY_ERRORS:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise
    else:
        edited_cfg = build_subject_noop_run_config(
            preset_data,
            model_id=norm_edt_id,
            adapter_name=str(eff_adapter),
            output_dir=str(Path(out) / "edited"),
            profile=profile_name,
            tier=tier_name,
            guards_order=guards_order,
        )
        edited_yaml = tmp_dir / "edited_noop.yaml"
        _dump_yaml(edited_yaml, edited_cfg)
        _info("Running edited (no-op, Compare & Evaluate)", tag="EXEC", emoji="🧪")
        _debug(f"Edited config: {edited_yaml}")
        from .run import run_command as _run

        with _suppress_child_output(verbosity == VERBOSITY_QUIET) as quiet_buffer:
            try:
                with timed_step(
                    console=console,
                    style=output_style,
                    timings=timings,
                    key="subject",
                    tag="EXEC",
                    message="Subject",
                    emoji="🧪",
                ):
                    edited_run_result = _run(
                        config=str(edited_yaml),
                        profile=profile_name,
                        out=str(Path(out) / "edited"),
                        tier=tier_name,
                        baseline=str(baseline_report_path),
                        device=device,
                        until_pass=False,
                        max_attempts=1,
                        timeout=None,
                        edit_label=subject_label,
                        style=output_style.name,
                        progress=progress,
                        timing=False,
                        allow_network=allow_network,
                        allow_host_execution=allow_host_execution,
                        allow_third_party_plugins=allow_third_party_plugins,
                        allow_remote_code=allow_remote_code,
                        prefer_local_files_only=prefer_local_files_only,
                        no_color=no_color,
                    )
            except typer.Exit:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise
            except _CHILD_RUN_REPLAY_ERRORS:
                if quiet_buffer is not None:
                    console.print(quiet_buffer.getvalue(), markup=False)
                raise

    try:
        edited_report = require_run_report_artifact(
            edited_run_result,
            stage="Edited",
        )
    except ConfigError as exc:
        _fail(str(getattr(exc, "message", exc)), exit_code=1)
    _debug(f"Edited report: {edited_report}")

    _phase(3, 3, "EVALUATION REPORT GENERATION")

    def _emit_evaluation_report() -> None:
        _info("Emitting evaluation report", tag="EXEC", emoji="📜")
        with timed_step(
            console=console,
            style=output_style,
            timings=timings,
            key="evaluation_report",
            tag="EXEC",
            message="Evaluation Report",
            emoji="📜",
        ):
            generate_reports(
                run=str(edited_report),
                format="report",
                baseline=str(baseline_report_path),
                output=str(report_out),
            )
        emit_runtime_manifest(
            Path(report_out) / "evaluation.report.json",
            config_payload={
                "command": "evaluate",
                "baseline": baseline,
                "subject": subject,
                "adapter": adapter,
                "profile": profile_name,
                "tier": tier_name,
                "preset": preset,
                "out": out,
                "report_out": report_out,
                "edit_config": edit_config,
                "edit_label": edit_label,
                "allow_network": allow_network,
                "allow_remote_code": allow_remote_code,
                "allow_third_party_plugins": allow_third_party_plugins,
            },
            extra={
                "command": "evaluate",
                "profile": profile_name,
                "tier": tier_name,
            },
        )

    # CI/Release hard‑abort: fail fast when primary metric is not computable.
    try:
        prof = str(profile or "").strip().lower()
    except _TEXT_NORMALIZATION_ERRORS:
        prof = ""
    if prof in {"ci", "ci_cpu", "release"}:
        try:
            with Path(edited_report).open("r", encoding="utf-8") as fh:
                edited_payload = json.load(fh)
        except _QUIET_REPORT_LOAD_ERRORS as exc:
            print_event(
                console,
                "FAIL",
                f"Failed to read edited report: {exc}",
                style=output_style,
                emoji="❌",
            )
            raise typer.Exit(1) from exc

        outcome = apply_edited_primary_metric_policy(
            edited_payload,
            profile=profile,
        )
        if outcome.diagnostic is not None and outcome.error is not None:
            print_event(
                console,
                "WARN",
                outcome.diagnostic.message,
                style=output_style,
                emoji="⚠️",
            )
            _emit_evaluation_report()
            raise typer.Exit(resolve_command_exit_code(outcome.error, profile=profile))

    _emit_evaluation_report()
    if timing:
        if total_start is not None:
            timings["total"] = max(0.0, float(perf_counter() - total_start))
        else:
            timings["total"] = (
                float(timings.get("baseline", 0.0))
                + float(timings.get("subject", 0.0))
                + float(timings.get("evaluation_report", 0.0))
            )
        print_timing_summary(
            console,
            timings,
            style=output_style,
            order=[
                ("Baseline", "baseline"),
                ("Subject", "subject"),
                ("Evaluation Report", "evaluation_report"),
                ("Total", "total"),
            ],
        )
    if verbosity == VERBOSITY_QUIET:
        _print_quiet_summary(
            report_out=Path(report_out),
            baseline=src_id,
            subject=edt_id,
            profile=profile_name,
        )
