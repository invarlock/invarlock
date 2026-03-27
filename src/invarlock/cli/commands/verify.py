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

from ...reporting.verify_contract import run_verify_reports
from .._json import emit as _emit_json

console = Console()


def verify_command(
    reports: list[Path],
    baseline: Path | None = None,
    tolerance: float = 1e-9,
    profile: str | None = "dev",
    json_out: bool = False,
    allow_unattested_artifacts: bool = False,
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
        allow_unattested_artifacts=bool(allow_unattested_artifacts),
        json_mode=bool(json_out),
    )
    if not json_out:
        for line in result.human_lines:
            console.print(line)
    if json_out:
        _emit_json(result.payload, result.exit_code)
    if result.exit_code != 0:
        raise SystemExit(result.exit_code)
