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

import typer
from rich.console import Console

from ...reporting.verify_contract import verify_reports_contract
from .._json import emit as _emit_json

console = Console()


def verify_command(
    reports: list[Path] = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="One or more evaluation report JSON files to verify.",
    ),
    baseline: Path | None = typer.Option(
        None,
        "--baseline",
        help="Optional baseline evaluation report (or run report) JSON to enforce provider parity.",
    ),
    tolerance: float = typer.Option(
        1e-9,
        "--tolerance",
        help="Tolerance for analysis-basis comparisons (mean log-loss).",
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
        help="Allow verification of reports that do not have container attestation.",
    ),
) -> None:
    """
    Verify evaluation report integrity.

    Ensures each evaluation report passes schema validation, ratio consistency checks,
    and strict pairing requirements (match=1.0, overlap=0.0).
    """

    try:
        from typer.models import OptionInfo as _OptionInfo  # type: ignore
    except Exception:  # pragma: no cover

        class _OptionInfo:  # type: ignore
            pass

    if isinstance(allow_unattested_artifacts, _OptionInfo):
        allow_unattested_artifacts = False
    exit_code, payload = verify_reports_contract(
        reports,
        baseline=baseline,
        tolerance=tolerance,
        profile=profile,
        allow_unattested_artifacts=bool(allow_unattested_artifacts),
        json_mode=bool(json_out),
        console_obj=console,
    )
    if json_out:
        _emit_json(payload, exit_code)
    if exit_code != 0:
        raise SystemExit(exit_code)
