from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError

import click
import typer
from rich.console import Console

from invarlock.cli import output as cli_output
from invarlock.cli.constants import RUNTIME_VERIFY_FORMAT_VERSION
from invarlock.runtime_verify import verify_runtime_manifest

_RUNTIME_VERIFY_HELP = (
    "Verify an evaluation report against its runtime.manifest.json companion."
)


def _emit_version(console: Console) -> None:
    try:
        from importlib.metadata import version as _package_version
    except (ImportError, ModuleNotFoundError, PackageNotFoundError):
        _package_version = None
    resolved = None
    if _package_version is not None:
        try:
            resolved = _package_version("invarlock")
        except PackageNotFoundError:
            resolved = None
    if not resolved:
        try:
            from invarlock import __version__ as resolved
        except (ImportError, ModuleNotFoundError):
            resolved = None
    if resolved:
        console.print(f"InvarLock runtime verifier {resolved}")
        return
    console.print("InvarLock runtime verifier version unknown")


def _runtime_verify_payload(*, report: str, manifest: str) -> dict[str, object]:
    result = verify_runtime_manifest(report, manifest)
    return {
        "format_version": RUNTIME_VERIFY_FORMAT_VERSION,
        "ok": result.ok,
        "errors": list(result.errors),
        "report": result.report,
        "manifest": result.manifest,
    }


def _run_runtime_verify(
    *,
    report: str,
    manifest: str,
    emit_json: bool,
    no_color: bool,
) -> int:
    payload = _runtime_verify_payload(report=report, manifest=manifest)
    if emit_json:
        print(json.dumps(payload, sort_keys=True))
        return 0 if bool(payload["ok"]) else 1

    console = cli_output.make_console(no_color=no_color)
    if bool(payload["ok"]):
        cli_output.print_command_event(
            console,
            "PASS",
            "Runtime manifest verification passed",
        )
    else:
        cli_output.print_command_event(
            console,
            "FAIL",
            "Runtime manifest verification failed",
        )
    cli_output.print_command_detail(console, f"Report: {payload['report']}")
    cli_output.print_command_detail(console, f"Manifest: {payload['manifest']}")
    for error in payload["errors"]:
        cli_output.print_command_detail(console, str(error), prefix="  -")
    return 0 if bool(payload["ok"]) else 1


def runtime_verify_callback(
    report: str = typer.Option(
        ...,
        "--report",
        help="Path to the canonical evaluation.report.json bundle.",
    ),
    manifest: str = typer.Option(
        ...,
        "--manifest",
        help="Path to the sibling runtime.manifest.json companion.",
    ),
    emit_json: bool = typer.Option(
        False,
        "--json",
        help="Emit a machine-readable JSON result.",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable ANSI colors (respects NO_COLOR=1).",
    ),
    show_version: bool = typer.Option(
        False,
        "--version",
        "-V",
        is_eager=True,
        help="Show runtime verifier version and exit.",
    ),
) -> None:
    if show_version:
        _emit_version(cli_output.make_console(no_color=no_color))
        raise typer.Exit(0)
    raise typer.Exit(
        _run_runtime_verify(
            report=report,
            manifest=manifest,
            emit_json=emit_json,
            no_color=no_color,
        )
    )


runtime_verify_app = typer.main.get_command_from_info(
    typer.main.CommandInfo(
        name="runtime-verify",
        callback=runtime_verify_callback,
        help=_RUNTIME_VERIFY_HELP,
        no_args_is_help=True,
    ),
    pretty_exceptions_short=True,
    rich_markup_mode="rich",
)


def build_click_command() -> click.Command:
    return runtime_verify_app


def main(argv: Sequence[str] | None = None) -> int:
    command = build_click_command()
    try:
        result = command.main(
            args=list(argv) if argv is not None else None,
            prog_name="invarlock advanced runtime-verify",
            standalone_mode=False,
        )
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
