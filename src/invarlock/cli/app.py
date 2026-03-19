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
    def list_commands(self, ctx):  # type: ignore[override]
        return [
            "evaluate",
            "calibrate",
            "report",
            "verify",
            "proof-pack",
            "policy",
            "run",
            "plugins",
            "doctor",
            "version",
        ]

    def get_command(self, ctx, cmd_name):  # type: ignore[override]
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        if _load_lazy_subapp(self, cmd_name):
            return super().get_command(ctx, cmd_name)
        return None


# Initialize CLI app
app = typer.Typer(
    name="invarlock",
    help=(
        "InvarLock — evaluate model changes with deterministic pairing and safety gates.\n"
        "Quick path: invarlock evaluate --baseline <MODEL> --subject <MODEL>\n"
        "Hint: use --edit-config to run the built-in quant_rtn demo.\n"
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
            from invarlock.reporting.report_builder import (
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

Order: evaluate → report → run → plugins → doctor → version
"""


@app.command(
    name="evaluate",
    help=(
        "Evaluate a subject model against a baseline and generate an evaluation report. "
        "Use when you have two model snapshots and want pass/fail gating."
    ),
)
def _evaluate_lazy(
    source: str = typer.Option(
        ..., "--source", "--baseline", help="Baseline model dir or Hub ID"
    ),
    edited: str = typer.Option(
        ..., "--edited", "--subject", help="Subject model dir or Hub ID"
    ),
    baseline_report: str | None = typer.Option(
        None,
        "--baseline-report",
        help=(
            "Reuse an existing baseline run report.json (skips baseline evaluation). "
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
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colors (respects NO_COLOR=1)"
    ),
):
    from .commands.evaluate import evaluate_command as _eval

    return _eval(
        source=source,
        edited=edited,
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
        no_color=no_color,
    )


def _register_subapps() -> None:
    # Keep single-command registration light; group-style subapps are loaded on
    # demand by OrderedGroup.get_command().
    from .commands.doctor import doctor_command as _doctor_cmd

    app.command(name="doctor")(_doctor_cmd)


def _load_lazy_subapp(group: TyperGroup, name: str) -> bool:
    def _register_lazy(name: str, subapp: typer.Typer) -> bool:
        command = typer.main.get_command(subapp)
        command.name = name
        group.add_command(command, name=name)
        return True

    if name == "report":
        from .commands.report import report_app as _report_app

        return _register_lazy(name, _report_app)
    if name == "policy":
        from .commands.policy import policy_app as _policy_app

        return _register_lazy(name, _policy_app)
    if name == "proof-pack":
        from .commands.proof_pack import proof_pack_app as _proof_pack_app

        return _register_lazy(name, _proof_pack_app)
    if name == "plugins":
        from .commands.plugins import plugins_app as _plugins_app

        return _register_lazy(name, _plugins_app)
    if name == "calibrate":
        try:
            from .commands.calibrate import calibrate_app as _calibrate_app
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised in venv test
            missing = getattr(exc, "name", "") or ""
            if missing in {"torch", "transformers"}:
                return False
            raise
        return _register_lazy(name, _calibrate_app)
    return False


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
    )


@app.command(
    name="run",
    help=(
        "Execute an end-to-end run from a YAML config (edit + guards + reports). "
        "Writes run artifacts and optionally an evaluation report."
    ),
)
def _run_typed(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to YAML configuration file"
    ),
    device: str | None = typer.Option(
        None, "--device", help="Device override (auto|cuda|mps|cpu)"
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="Profile to apply (ci|release)"
    ),
    out: str | None = typer.Option(None, "--out", help="Output directory override"),
    edit: str | None = typer.Option(
        None,
        "--edit",
        help="Edit name override (canonical plugin name, e.g. quant_rtn)",
    ),
    edit_label: str | None = typer.Option(
        None,
        "--edit-label",
        help=(
            "Edit algorithm label for BYOE models. Use 'noop' for baseline, "
            "'quant_rtn' etc. for built-in edits, 'custom' for pre-edited models."
        ),
    ),
    tier: str | None = typer.Option(
        None,
        "--tier",
        help="Auto-tuning tier override (conservative|balanced|aggressive)",
    ),
    metric_kind: str | None = typer.Option(
        None,
        "--metric-kind",
        help="Primary metric kind override (ppl_causal|ppl_mlm|accuracy|etc.)",
    ),
    probes: int | None = typer.Option(
        None, "--probes", help="Number of micro-probes (0=deterministic, >0=adaptive)"
    ),
    until_pass: bool = typer.Option(
        False,
        "--until-pass",
        help="Retry until evaluation report passes gates (max 3 attempts)",
    ),
    max_attempts: int = typer.Option(
        3, "--max-attempts", help="Maximum retry attempts for --until-pass mode"
    ),
    timeout: int | None = typer.Option(
        None, "--timeout", help="Timeout in seconds for --until-pass mode"
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help="Path to baseline report.json for evaluation report validation",
    ),
    no_cleanup: bool = typer.Option(
        False, "--no-cleanup", help="Skip cleanup of temporary artifacts"
    ),
    style: str | None = typer.Option(
        None, "--style", help="Output style (audit|friendly)"
    ),
    progress: bool = typer.Option(
        False, "--progress", help="Show progress done messages"
    ),
    timing: bool = typer.Option(False, "--timing", help="Show timing summary"),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colors (respects NO_COLOR=1)"
    ),
):
    from .commands.run import run_command as _run

    return _run(
        config=config,
        device=device,
        profile=profile,
        out=out,
        edit=edit,
        edit_label=edit_label,
        tier=tier,
        metric_kind=metric_kind,
        probes=probes,
        until_pass=until_pass,
        max_attempts=max_attempts,
        timeout=timeout,
        baseline=baseline,
        no_cleanup=no_cleanup,
        style=style,
        progress=progress,
        timing=timing,
        no_color=no_color,
    )


_register_subapps()


def main() -> None:
    """Main entry point for the InvarLock CLI."""
    enforce_default_security()
    app()


if __name__ == "__main__":
    main()
