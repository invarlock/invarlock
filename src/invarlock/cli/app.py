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

from invarlock.security import enforce_default_security

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
            "certify",
            "calibrate",
            "report",
            "verify",
            "run",
            "plugins",
            "doctor",
            "version",
        ]


# Initialize CLI app
app = typer.Typer(
    name="invarlock",
    help=(
        "InvarLock — certify model changes with deterministic pairing and safety gates.\n"
        "Quick path: invarlock certify --baseline <MODEL> --subject <MODEL>\n"
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


@app.command()
def version():
    """Show InvarLock version."""
    # Prefer package metadata when available so CLI reflects wheel truth
    try:
        from importlib.metadata import version as _pkg_version

        schema = None
        try:
            from invarlock.reporting.certificate import (
                CERTIFICATE_SCHEMA_VERSION as _SCHEMA,
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


"""Register command modules and groups in the desired help order.

Order: certify → report → run → plugins → doctor → version
"""


@app.command(
    name="certify",
    help=(
        "Certify a subject model against a baseline and generate a safety certificate. "
        "Use when you have two model snapshots and want pass/fail gating."
    ),
)
def _certify_lazy(
    source: str = typer.Option(
        ..., "--source", "--baseline", help="Baseline model dir or Hub ID"
    ),
    edited: str = typer.Option(
        ..., "--edited", "--subject", help="Subject model dir or Hub ID"
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
    cert_out: str = typer.Option(
        "reports/cert", "--cert-out", help="Certificate output directory"
    ),
    edit_config: str | None = typer.Option(
        None, "--edit-config", help="Edit preset to apply a demo edit (quant_rtn)"
    ),
):
    from .commands.certify import certify_command as _cert

    return _cert(
        source=source,
        edited=edited,
        adapter=adapter,
        device=device,
        profile=profile,
        tier=tier,
        preset=preset,
        out=out,
        cert_out=cert_out,
        edit_config=edit_config,
    )


def _register_subapps() -> None:
    # Import sub-apps lazily to keep module import light and satisfy E402
    from .commands.doctor import doctor_command as _doctor_cmd
    from .commands.plugins import plugins_app as _plugins_app
    from .commands.report import report_app as _report_app

    # Always-available subapps (lightweight imports)
    app.add_typer(_report_app, name="report")
    app.add_typer(_plugins_app, name="plugins")
    app.command(name="doctor")(_doctor_cmd)

    # Optional: calibration subapp. This transitively imports guards, which may
    # depend on torch/transformers. In minimal environments (no heavy deps),
    # skip registration so `python -m invarlock --help` stays import-safe.
    try:
        from .commands.calibrate import calibrate_app as _calibrate_app
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in venv test
        missing = getattr(exc, "name", "") or ""
        if missing in {"torch", "transformers"}:
            return
        raise
    else:
        app.add_typer(_calibrate_app, name="calibrate")


@app.command(
    name="verify",
    help=(
        "Verify certificate JSON(s) against schema, pairing math, and gates. "
        "Use --json for a single-line machine-readable envelope."
    ),
)
def _verify_typed(
    certificates: list[str] = typer.Argument(
        ..., help="One or more certificate JSON files to verify."
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help="Optional baseline certificate/report JSON to enforce provider parity.",
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

    cert_paths = [_Path(c) for c in certificates]
    baseline_path = _Path(baseline) if isinstance(baseline, str) else None
    return _verify(
        certificates=cert_paths,
        baseline=baseline_path,
        tolerance=tolerance,
        profile=profile,
        json_out=json_out,
    )


@app.command(
    name="run",
    help=(
        "Execute an end-to-end run from a YAML config (edit + guards + reports). "
        "Writes run artifacts and optionally a safety certificate."
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
    edit: str | None = typer.Option(None, "--edit", help="Edit kind (quant|mixed)"),
    tier: str | None = typer.Option(
        None,
        "--tier",
        help="Auto-tuning tier override (conservative|balanced|aggressive)",
    ),
    probes: int | None = typer.Option(
        None, "--probes", help="Number of micro-probes (0=deterministic, >0=adaptive)"
    ),
    until_pass: bool = typer.Option(
        False, "--until-pass", help="Retry until certificate passes (max 3 attempts)"
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
        help="Path to baseline report.json for certificate validation",
    ),
    no_cleanup: bool = typer.Option(
        False, "--no-cleanup", help="Skip cleanup of temporary artifacts"
    ),
):
    from .commands.run import run_command as _run

    return _run(
        config=config,
        device=device,
        profile=profile,
        out=out,
        edit=edit,
        tier=tier,
        probes=probes,
        until_pass=until_pass,
        max_attempts=max_attempts,
        timeout=timeout,
        baseline=baseline,
        no_cleanup=no_cleanup,
    )


_register_subapps()


def main() -> None:
    """Main entry point for the InvarLock CLI."""
    enforce_default_security()
    app()


if __name__ == "__main__":
    main()
