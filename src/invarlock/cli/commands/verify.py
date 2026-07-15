"""
invarlock verify command
====================

Validates generated evaluation reports for internal consistency. The command
ensures schema compliance, checks that the primary metric ratio agrees with the
baseline reference, and enforces paired-window guarantees (match=1.0,
overlap=0.0). Strict paired PPL verification also independently replays the
reported confidence interval from independently supplied raw baseline windows.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any, TypedDict

import click
import typer

from invarlock.core.assurance_contract import normalize_verify_assurance_mode
from invarlock.core.exceptions import resolve_command_exit_code
from invarlock.runtime_verify import verify_runtime_manifest

from ...reporting.verify_contract import (
    VerifyDiagnostic,
    VerifyExecutionResult,
    VerifyOutcome,
    run_verify_reports,
)
from .. import output as cli_output
from ..constants import RUNTIME_VERIFY_FORMAT_VERSION

console = cli_output.make_console()
_RUNTIME_VERIFY_HELP = (
    "Verify an evaluation report against its runtime.manifest.json companion."
)


class RuntimeVerifyPayload(TypedDict):
    format_version: str
    ok: bool
    errors: list[str]
    report: str
    manifest: str
    binding_verified: bool
    expected_digest_matched: bool
    trust_status: str
    declared_image_digest: str | None


def _allow_unverified_provenance_for_runtime_provenance(
    runtime_provenance: str,
) -> bool:
    normalized_runtime_provenance = str(runtime_provenance or "").strip().lower()
    if normalized_runtime_provenance == "container":
        return False
    if normalized_runtime_provenance == "host":
        return True
    raise ValueError("Runtime provenance must be one of: container, host.")


def _render_verify_diagnostic(diagnostic: VerifyDiagnostic) -> None:
    level = str(diagnostic.level or "").lower()
    message = diagnostic.message
    if level == "pass":
        cli_output.print_command_event(console, "PASS", message)
        return
    if level == "fail":
        cli_output.print_command_event(console, "FAIL", message)
        return
    if level == "detail":
        cli_output.print_command_detail(console, message)
        return
    if level == "warning":
        cli_output.print_command_event(console, "WARN", message)
        return
    if level == "error":
        cli_output.print_command_event(console, "FAIL", message)
        return
    cli_output.print_command_detail(console, message, prefix="  ·")


def _verify_exit_code(
    result: VerifyExecutionResult,
    *,
    profile: str | None,
) -> int:
    if result.outcome == VerifyOutcome.OK:
        return 0
    if isinstance(result.error, Exception):
        return resolve_command_exit_code(result.error, profile=profile)
    if result.outcome == VerifyOutcome.MALFORMED:
        return 2
    return 1


def verify_command(
    reports: list[Path],
    baseline: Path | None = None,
    policy_pack: Path | None = None,
    tolerance: float = 1e-9,
    profile: str | None = None,
    json_out: bool = False,
    runtime_provenance: str = "container",
    assurance: str = "report",
    warning_policy: str = "pass",
    expected_runtime_image_digest: str | None = None,
) -> None:
    """
    Verify evaluation report integrity.

    Ensures each evaluation report passes schema validation, ratio consistency checks,
    and strict pairing requirements (match=1.0, overlap=0.0). Strict paired PPL
    reports require the external canonical noop baseline run report and a
    independently supplied policy pack.
    PPL intervals and accuracy counts are replayed from its raw evidence, and
    model/dataset/provider/tokenizer provenance must match the subject.
    """
    result = run_verify_reports(
        reports,
        baseline=baseline,
        policy_pack=policy_pack,
        tolerance=tolerance,
        profile=profile,
        allow_unverified_provenance=(
            _allow_unverified_provenance_for_runtime_provenance(runtime_provenance)
        ),
        json_mode=bool(json_out),
        assurance_mode=normalize_verify_assurance_mode(assurance),
        warning_policy=warning_policy,
        expected_runtime_image_digest=expected_runtime_image_digest,
    )
    exit_code = _verify_exit_code(result, profile=profile)
    if not json_out:
        for diagnostic in result.diagnostics:
            _render_verify_diagnostic(diagnostic)
    if json_out:
        payload = dict(result.payload)
        if result.include_resolution:
            payload["resolution"] = {"exit_code": exit_code}
        cli_output.emit(payload, exit_code)
    if exit_code != 0:
        raise SystemExit(exit_code)


def _emit_version(version_console: Any) -> None:
    package_version: Callable[[str], str] | None
    try:
        from importlib.metadata import version as package_version
    except (ImportError, ModuleNotFoundError, PackageNotFoundError):
        package_version = None
    resolved = None
    if package_version is not None:
        try:
            resolved = package_version("invarlock")
        except PackageNotFoundError:
            resolved = None
    if not resolved:
        try:
            from invarlock import __version__ as resolved
        except (ImportError, ModuleNotFoundError):
            resolved = None
    if resolved:
        version_console.print(f"InvarLock runtime verifier {resolved}")
        return
    version_console.print("InvarLock runtime verifier version unknown")


def _runtime_verify_payload(
    *,
    report: str,
    manifest: str,
    expected_runtime_image_digest: str | None = None,
) -> RuntimeVerifyPayload:
    if expected_runtime_image_digest is None:
        result = verify_runtime_manifest(report, manifest)
    else:
        result = verify_runtime_manifest(
            report,
            manifest,
            expected_image_digest=expected_runtime_image_digest,
        )
    return {
        "format_version": RUNTIME_VERIFY_FORMAT_VERSION,
        "ok": result.ok,
        "errors": list(result.errors),
        "report": result.report,
        "manifest": result.manifest,
        "binding_verified": result.binding_verified,
        "expected_digest_matched": result.expected_digest_matched,
        "trust_status": result.trust_status,
        "declared_image_digest": result.declared_image_digest,
    }


def _run_runtime_verify(
    *,
    report: str,
    manifest: str,
    emit_json: bool,
    no_color: bool,
    expected_runtime_image_digest: str | None = None,
) -> int:
    payload = _runtime_verify_payload(
        report=report,
        manifest=manifest,
        expected_runtime_image_digest=expected_runtime_image_digest,
    )
    if emit_json:
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0 if bool(payload["ok"]) else 1

    runtime_console = cli_output.make_console(no_color=no_color)
    if bool(payload["ok"]):
        cli_output.print_command_event(
            runtime_console,
            "PASS",
            "Runtime report/manifest binding passed",
        )
    else:
        cli_output.print_command_event(
            runtime_console,
            "FAIL",
            "Runtime manifest verification failed",
        )
    cli_output.print_command_detail(runtime_console, f"Report: {payload['report']}")
    cli_output.print_command_detail(runtime_console, f"Manifest: {payload['manifest']}")
    cli_output.print_command_detail(
        runtime_console, f"Runtime trust: {payload['trust_status']}"
    )
    for error in payload["errors"]:
        cli_output.print_command_detail(runtime_console, str(error), prefix="  -")
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
    expected_runtime_image_digest: str | None = typer.Option(
        None,
        "--expected-runtime-image-digest",
        help=("Independent sha256:... trust anchor for the declared runtime image."),
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
            expected_runtime_image_digest=expected_runtime_image_digest,
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
