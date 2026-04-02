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
from enum import Enum

import click
import typer
from rich.console import Console
from typer.core import TyperGroup

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


class ExecutionMode(str, Enum):
    ATTESTED = "attested"
    LOCAL = "local"


# Initialize CLI app
app = typer.Typer(
    name="invarlock",
    help=(
        "InvarLock — evaluate model changes with deterministic pairing and safety gates.\n"
        "Core path: invarlock evaluate --baseline <MODEL> --subject <MODEL>\n"
        "Then: invarlock verify <REPORT> and invarlock report html -i <REPORT> -o <HTML>\n"
        "Advanced workflows live under: invarlock advanced\n"
        "Tip: enable downloads with INVARLOCK_ALLOW_NETWORK=1 when fetching.\n"
        "Exit codes:\n"
        "  0=success\n"
        "  1=generic failure\n"
        "  2=schema invalid\n"
        "  3=hard abort ([INVARLOCK:EXXX])."
    ),
    no_args_is_help=True,
    cls=OrderedGroup,
)

console = Console()


def _emit_version() -> None:
    """Emit the InvarLock version string."""
    # Prefer package metadata when available so CLI reflects wheel truth
    try:
        from importlib.metadata import version as _pkg_version

        schema = None
        try:
            from invarlock.reporting.report_schema import (
                REPORT_SCHEMA_VERSION as _SCHEMA,
            )

            schema = _SCHEMA
        except Exception:
            schema = None
        msg = f"InvarLock {_pkg_version('invarlock')}"
        if schema:
            msg += f" · schema={schema}"
        console.print(msg)
        return
    except Exception:
        pass
    try:
        from invarlock import __version__

        console.print(f"InvarLock {__version__}")
    except Exception:
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
    adapter: str = typer.Option(
        "auto", "--adapter", help="Adapter name or 'auto' to resolve"
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
            "based on adapter)"
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
    progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show progress done messages"
    ),
    mode: ExecutionMode = typer.Option(
        ExecutionMode.ATTESTED,
        "--mode",
        help="Execution mode for model-loading steps.",
        case_sensitive=False,
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colors (respects NO_COLOR=1)"
    ),
    allow_network: bool = typer.Option(
        False,
        "--allow-network",
        help="Allow network access, including runtime-image pulls and model fetches.",
    ),
):
    from .commands.evaluate import evaluate_command as _eval

    return _eval(
        baseline=baseline,
        subject=subject,
        baseline_report=baseline_report,
        adapter=adapter,
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
        progress=progress,
        mode=mode.value,
        no_color=no_color,
        allow_network=allow_network,
    )


def _register_subapps() -> None:
    # Keep single-command registration light; group-style subapps are loaded on
    # demand by OrderedGroup.get_command().
    pass


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
        from .commands.advanced import advanced_app as _advanced_app

        return _register_lazy(name, _advanced_app)
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
        ..., help="One or more evaluation report JSON files to verify."
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help="Optional baseline evaluation report JSON to enforce provider parity.",
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
    allow_unattested_artifacts: bool = typer.Option(
        False,
        "--allow-unattested-artifacts",
        help="Allow verification of reports without runtime attestation metadata.",
    ),
):
    from pathlib import Path as _Path

    from .commands.verify import verify_command as _verify

    report_paths = [_Path(p) for p in reports]
    baseline_path = _Path(baseline) if isinstance(baseline, str) else None
    return _verify(
        reports=report_paths,
        baseline=baseline_path,
        tolerance=tolerance,
        profile=profile,
        json_out=json_out,
        allow_unattested_artifacts=allow_unattested_artifacts,
    )


_register_subapps()


def main() -> None:
    """Main entry point for the InvarLock CLI."""
    enforce_default_security()
    app()


if __name__ == "__main__":
    main()
