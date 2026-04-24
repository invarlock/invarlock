"""
invarlock verify command
====================

Validates generated evaluation reports for internal consistency. The command
ensures schema compliance, checks that the primary metric ratio agrees with the
baseline reference, and enforces paired-window guarantees (match=1.0,
overlap=0.0).
"""

from __future__ import annotations

from pathlib import Path

from invarlock.exit_codes import resolve_command_exit_code

from ...reporting.verify_contract import (
    VerifyDiagnostic,
    VerifyExecutionResult,
    VerifyOutcome,
    run_verify_reports,
)
from .. import output as cli_output
from .._json import emit as _emit_json

console = cli_output.make_console()


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
    tolerance: float = 1e-9,
    profile: str | None = "dev",
    json_out: bool = False,
    runtime_provenance: str = "container",
) -> None:
    """
    Verify evaluation report integrity.

    Ensures each evaluation report passes schema validation, ratio consistency checks,
    and strict pairing requirements (match=1.0, overlap=0.0).
    """
    result = run_verify_reports(
        reports,
        baseline=baseline,
        tolerance=tolerance,
        profile=profile,
        allow_unverified_provenance=(
            _allow_unverified_provenance_for_runtime_provenance(runtime_provenance)
        ),
        json_mode=bool(json_out),
    )
    exit_code = _verify_exit_code(result, profile=profile)
    if not json_out:
        for diagnostic in result.diagnostics:
            _render_verify_diagnostic(diagnostic)
    if json_out:
        payload = dict(result.payload)
        if result.include_resolution:
            payload["resolution"] = {"exit_code": exit_code}
        _emit_json(payload, exit_code)
    if exit_code != 0:
        raise SystemExit(exit_code)
