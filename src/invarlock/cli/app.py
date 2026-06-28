"""
InvarLock CLI Main Entry Point (unified namespace)
=============================================

Modern CLI with clean command interface using modular command structure.

Import guard: set `INVARLOCK_LIGHT_IMPORT=1` to avoid heavy plugin discovery and
third‑party imports during docs/tests. This keeps `import invarlock.cli.app` safe in
minimal environments.
"""

from __future__ import annotations

import os
from enum import StrEnum
from importlib.metadata import PackageNotFoundError

import click
import typer
from rich.console import Console
from typer.core import TyperGroup

from invarlock.core.report_inputs import (
    ReportInputError,
    resolve_report_input_path,
)
from invarlock.security import (
    enforce_default_security,
    enforce_network_policy,
    network_policy_allows,
)

# Lightweight import mode disables heavy side effects in some modules, but we no
# longer force plugin discovery off globally here; individual commands may gate
# discovery based on their own flags.
LIGHT_IMPORT = os.getenv("INVARLOCK_LIGHT_IMPORT", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


class ExecutionMode(StrEnum):
    CONTAINER = "container"
    HOST = "host"


class RuntimeProvenanceMode(StrEnum):
    CONTAINER = "container"
    HOST = "host"


# Deterministic help ordering
class OrderedGroup(TyperGroup):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return [
            "evaluate",
            "report",
            "verify",
            "doctor",
            "advanced",
            "version",
        ]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        if _load_lazy_subapp(self, cmd_name):
            return super().get_command(ctx, cmd_name)
        return None


class AdvancedGroup(TyperGroup):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return ["evidence-pack", "policy", "plugins", "calibrate", "runtime-verify"]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        if _load_advanced_subapp(self, cmd_name):
            return super().get_command(ctx, cmd_name)
        return None


# Initialize CLI app
app = typer.Typer(
    name="invarlock",
    help=(
        "InvarLock — evaluate model changes with deterministic pairing and safety gates.\n"
        "Core path: invarlock evaluate --baseline <MODEL> --subject <MODEL>\n"
        "Then: invarlock verify <REPORT> and invarlock report html -i <REPORT> -o <HTML>\n"
        "Advanced workflows live under: invarlock advanced\n"
        "Tip: enable downloads with --allow-network when fetching.\n"
        "Exit codes:\n"
        "  0=success\n"
        "  1=generic failure\n"
        "  2=schema invalid\n"
        "  3=hard abort with structured error code."
    ),
    no_args_is_help=True,
    cls=OrderedGroup,
)

console = Console()
advanced_app = typer.Typer(
    help=(
        "Advanced and maintenance workflows. "
        "These commands are intentionally outside the core evaluate/verify/report path."
    ),
    no_args_is_help=True,
    cls=AdvancedGroup,
)
_VERSION_IMPORT_ERRORS = (
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    PackageNotFoundError,
)


def _resolve_schema_version() -> str | None:
    try:
        from invarlock.reporting.report_schema import REPORT_SCHEMA_VERSION
    except _VERSION_IMPORT_ERRORS:
        return None
    return (
        REPORT_SCHEMA_VERSION
        if isinstance(REPORT_SCHEMA_VERSION, str) and REPORT_SCHEMA_VERSION
        else None
    )


def _resolve_package_version() -> str | None:
    try:
        from importlib.metadata import version as _pkg_version
    except _VERSION_IMPORT_ERRORS:
        return None
    try:
        resolved = _pkg_version("invarlock")
    except _VERSION_IMPORT_ERRORS:
        return None
    return resolved if isinstance(resolved, str) and resolved else None


def _resolve_module_version() -> str | None:
    try:
        from invarlock import __version__
    except _VERSION_IMPORT_ERRORS:
        return None
    return __version__ if isinstance(__version__, str) and __version__ else None


def _emit_version() -> None:
    """Emit the InvarLock version string."""
    # Prefer package metadata when available so CLI reflects wheel truth
    package_version = _resolve_package_version()
    if package_version is not None:
        msg = f"InvarLock {package_version}"
        schema = _resolve_schema_version()
        if schema:
            msg += f" · schema={schema}"
        console.print(msg)
        return
    module_version = _resolve_module_version()
    if module_version is not None:
        console.print(f"InvarLock {module_version}")
        return
    console.print("InvarLock version unknown")


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    show_version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        is_eager=True,
    ),
) -> None:
    was_allowed = network_policy_allows()
    enforce_default_security()
    ctx.call_on_close(lambda: enforce_network_policy(was_allowed))
    if show_version:
        _emit_version()
        raise typer.Exit()


@app.command()
def version():
    """Show InvarLock version."""
    _emit_version()


"""Register command modules and groups in the desired help order.

Order: evaluate → report → verify → doctor → advanced → version
"""


@app.command(
    name="evaluate",
    help=(
        "Evaluate a subject model against a baseline and generate an evaluation report. "
        "Use when you have two model snapshots and want pass/fail gating."
    ),
)
def _evaluate_lazy(
    baseline: str = typer.Option(
        ..., "--baseline", help="Baseline model dir or Hub ID"
    ),
    subject: str = typer.Option(..., "--subject", help="Subject model dir or Hub ID"),
    baseline_report: str | None = typer.Option(
        None,
        "--baseline-report",
        help=(
            "Reuse an existing baseline run report.json file (explicit path; skips baseline evaluation). "
            "Must include stored evaluation windows (e.g., set INVARLOCK_STORE_EVAL_WINDOWS=1)."
        ),
    ),
    baseline_adapter: str = typer.Option(
        "auto",
        "--baseline-adapter",
        help="Adapter for the baseline side, or 'auto' to resolve from baseline.",
    ),
    subject_adapter: str = typer.Option(
        "auto",
        "--subject-adapter",
        help="Adapter for the subject side, or 'auto' to resolve from subject.",
    ),
    device: str | None = typer.Option(
        None, "--device", help="Device override for runs (auto|cuda|mps|cpu)"
    ),
    profile: str = typer.Option("ci", "--profile", help="Profile (ci|release)"),
    tier: str = typer.Option("balanced", "--tier", help="Tier label for context"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help=(
            "Universal preset path to use (defaults to causal or masked preset "
            "based on the subject adapter)"
        ),
    ),
    out: str = typer.Option("runs", "--out", help="Base output directory"),
    report_out: str = typer.Option(
        "reports/eval", "--report-out", help="Evaluation report output directory"
    ),
    edit_config: str | None = typer.Option(
        None, "--edit-config", help="Edit preset to apply a demo edit (quant_rtn)"
    ),
    edit_label: str | None = typer.Option(
        None,
        "--edit-label",
        help=(
            "Edit algorithm label for BYOE models. Use 'noop' for baseline, "
            "'quant_rtn' etc. for built-in edits, 'custom' for pre-edited models."
        ),
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Minimal output (suppress run/report detail)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose output (include debug details)"
    ),
    banner: bool = typer.Option(
        True, "--banner/--no-banner", help="Show header banner"
    ),
    style: str = typer.Option("audit", "--style", help="Output style (audit|friendly)"),
    timing: bool = typer.Option(False, "--timing", help="Show timing summary"),
    timing_json: str | None = typer.Option(
        None,
        "--timing-json",
        help="Write machine-readable evaluate timing data to this JSON path.",
    ),
    defer_report_rendering: bool = typer.Option(
        False,
        "--defer-report-rendering",
        help=(
            "Write JSON evidence sidecars only; skip markdown/evidence bundle "
            "rendering in the hot path."
        ),
    ),
    progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show progress done messages"
    ),
    execution_mode: ExecutionMode = typer.Option(
        ExecutionMode.CONTAINER,
        "--execution-mode",
        help="Execution mode for evaluation (container|host).",
        case_sensitive=False,
    ),
    assurance: str = typer.Option(
        "strict",
        "--assurance",
        help="Assurance mode for evaluation (strict|off).",
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colors (respects NO_COLOR=1)"
    ),
    allow_network: bool = typer.Option(
        False,
        "--allow-network",
        help="Allow network access, including runtime-image pulls and model fetches.",
    ),
    allow_remote_code: bool = typer.Option(
        False,
        "--allow-remote-code",
        help="Allow trust_remote_code-style model loading for this command.",
    ),
):
    from .commands.evaluate import evaluate_command as _eval

    return _eval(
        baseline=baseline,
        subject=subject,
        baseline_report=baseline_report,
        baseline_adapter=baseline_adapter,
        subject_adapter=subject_adapter,
        device=device,
        profile=profile,
        tier=tier,
        preset=preset,
        out=out,
        report_out=report_out,
        edit_config=edit_config,
        edit_label=edit_label,
        quiet=quiet,
        verbose=verbose,
        banner=banner,
        style=style,
        timing=timing,
        timing_json=timing_json,
        defer_report_rendering=defer_report_rendering,
        progress=progress,
        execution_mode=execution_mode.value,
        assurance=assurance,
        no_color=no_color,
        allow_network=allow_network,
        allow_remote_code=allow_remote_code,
    )


def _register_subapps() -> None:
    # Keep single-command registration light; group-style subapps are loaded on
    # demand by OrderedGroup.get_command().
    pass


@advanced_app.callback(invoke_without_command=True)
def _advanced_root() -> None:
    """Advanced command namespace."""


def _missing_dependency_subapp(name: str, missing: str) -> typer.Typer:
    subapp = typer.Typer(help=f"{name} requires optional dependency {missing!r}.")

    @subapp.callback(invoke_without_command=True)
    def _missing() -> None:
        raise click.UsageError(
            f"`invarlock advanced {name}` requires optional dependency {missing!r}."
        )

    return subapp


def _load_advanced_subapp(group: TyperGroup, name: str) -> bool:
    def _register(sub_name: str, subapp: typer.Typer) -> bool:
        command = typer.main.get_command(subapp)
        command.name = sub_name
        group.add_command(command, name=sub_name)
        return True

    def _register_command(sub_name: str, command: click.Command) -> bool:
        command.name = sub_name
        group.add_command(command, name=sub_name)
        return True

    if name == "evidence-pack":
        from .commands.evidence_pack import evidence_pack_app

        return _register(name, evidence_pack_app)
    if name == "policy":
        from .commands.policy import policy_app

        return _register(name, policy_app)
    if name == "plugins":
        from .commands.plugins import plugins_app

        return _register(name, plugins_app)
    if name == "calibrate":
        try:
            from .commands.calibrate import calibrate_app
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised in venvs
            missing = getattr(exc, "name", "") or "optional runtime"
            return _register(name, _missing_dependency_subapp(name, missing))
        return _register(name, calibrate_app)
    if name == "runtime-verify":
        from invarlock.cli.commands.verify import runtime_verify_app

        return _register_command(name, runtime_verify_app)
    return False


def _load_lazy_subapp(group: TyperGroup, name: str) -> bool:
    def _register_lazy(name: str, subapp: typer.Typer) -> bool:
        command = typer.main.get_command(subapp)
        command.name = name
        group.add_command(command, name=name)
        return True

    if name == "report":
        from .commands.report import report_app as _report_app

        return _register_lazy(name, _report_app)
    if name == "advanced":
        return _register_lazy(name, advanced_app)
    return False


@app.command(
    name="doctor",
    help=(
        "Inspect runtime health, optional dependencies, datasets, and explicit report inputs. "
        "Optional report paths must be explicit report.json or evaluation.report.json files."
    ),
)
def _doctor_typed(
    config: str | None = typer.Option(
        None, "--config", help="Optional config file to validate and inspect."
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="Optional execution profile to validate."
    ),
    baseline: str | None = typer.Option(
        None, "--baseline", help="Optional baseline model path or id for quick checks."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    tier: str | None = typer.Option(
        None, "--tier", help="Optional tier context for config validation."
    ),
    baseline_report: str | None = typer.Option(
        None,
        "--baseline-report",
        help="Explicit baseline report.json or evaluation.report.json path for cross-checks.",
    ),
    subject_report: str | None = typer.Option(
        None,
        "--subject-report",
        help="Explicit subject report.json or evaluation.report.json path for cross-checks.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Return a non-zero exit code on warnings as well as errors.",
    ),
):
    from .commands.doctor import doctor_command as _doctor

    return _doctor(
        config=config,
        profile=profile,
        baseline=baseline,
        json_out=json_out,
        tier=tier,
        baseline_report=baseline_report,
        subject_report=subject_report,
        strict=strict,
    )


@app.command(
    name="verify",
    help=(
        "Verify evaluation report JSON(s) against schema, pairing math, and gates. "
        "Use --json for a single-line machine-readable envelope."
    ),
)
def _verify_typed(
    reports: list[str] = typer.Argument(
        ...,
        help=(
            "One or more evaluation report JSON files or directories containing "
            "canonical evaluation.report.json to verify."
        ),
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help=(
            "Optional baseline report JSON file or directory containing canonical "
            "report.json or evaluation.report.json to enforce provider parity."
        ),
    ),
    tolerance: float = typer.Option(
        1e-9, "--tolerance", help="Tolerance for analysis-basis comparisons."
    ),
    profile: str | None = typer.Option(
        "dev",
        "--profile",
        help="Execution profile affecting parity enforcement and exit codes (dev|ci|release).",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON (suppresses human-readable output)",
    ),
    runtime_provenance: RuntimeProvenanceMode = typer.Option(
        RuntimeProvenanceMode.CONTAINER,
        "--runtime-provenance",
        help="Runtime provenance mode for verification (container|host).",
    ),
    assurance: str = typer.Option(
        "report",
        "--assurance",
        help="Assurance verification mode (report|strict|off).",
    ),
    warning_policy: str = typer.Option(
        "pass",
        "--warning-policy",
        help="Guard-warning handling mode (pass|fail).",
    ),
    fail_on_warnings: bool = typer.Option(
        False,
        "--fail-on-warnings",
        help="Alias for --warning-policy fail.",
    ),
):
    from pathlib import Path as _Path

    from .commands.verify import verify_command as _verify

    try:
        report_paths = [
            resolve_report_input_path(_Path(p), expected_kind="evaluation")
            for p in reports
        ]
        baseline_path = (
            resolve_report_input_path(_Path(baseline), expected_kind="any")
            if isinstance(baseline, str)
            else None
        )
    except ReportInputError as exc:
        console.print(f"FAIL {exc}")
        raise typer.Exit(2) from exc
    return _verify(
        reports=report_paths,
        baseline=baseline_path,
        tolerance=tolerance,
        profile=profile,
        json_out=json_out,
        runtime_provenance=runtime_provenance.value,
        assurance=assurance,
        warning_policy="fail" if fail_on_warnings else warning_policy,
    )


_register_subapps()


def main() -> None:
    """Main entry point for the InvarLock CLI."""
    enforce_default_security()
    app()


if __name__ == "__main__":
    main()
