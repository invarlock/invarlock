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

from rich.console import Console

from invarlock.exit_codes import resolve_command_exit_code

from ...reporting.verify_contract import (
    VerifyDiagnostic,
    VerifyExecutionResult,
    VerifyOutcome,
    run_verify_reports,
)
from .._json import emit as _emit_json

console = Console()


def _allow_unattested_artifacts_for_assurance(assurance: str) -> bool:
    normalized_assurance = str(assurance or "").strip().lower()
    if normalized_assurance == "attested":
        return False
    if normalized_assurance == "trusted-local":
        return True
    raise ValueError("Assurance level must be one of: attested, trusted-local.")


def _render_verify_diagnostic(diagnostic: VerifyDiagnostic) -> None:
    level = str(diagnostic.level or "").lower()
    message = diagnostic.message
    if level == "pass":
        console.print(f"[green]PASS[/green] {message}")
        return
    if level == "fail":
        console.print(f"[red]FAIL[/red] {message}")
        return
    if level == "detail":
        console.print(f"  ↳ {message}")
        return
    if level == "warning":
        console.print(f"[yellow]⚠️  {message}[/yellow]")
        return
    if level == "error":
        console.print(f"[red]❌ {message}[/red]")
        return
    console.print(message)


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
    assurance: str = "attested",
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
        allow_unattested_artifacts=_allow_unattested_artifacts_for_assurance(assurance),
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
