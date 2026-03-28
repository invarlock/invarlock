"""Shell-level artifact persistence helpers for run execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from invarlock.cli.run_shell_output import _event


def emit_run_artifacts(
    *, report: Any, out_dir: Path, filename_prefix: str, console: Console
) -> dict[str, str]:
    """Save run report and return emitted artifact paths."""
    from invarlock.reporting.report_files import save_report

    _event(console, "DATA", "Saving run report...", emoji="💾")
    return save_report(
        report, out_dir, formats=["json"], filename_prefix=filename_prefix
    )


def postprocess_and_summarize(
    *,
    report: dict[str, Any],
    run_dir: Path,
    run_config: Any,
    console: Console,
) -> dict[str, str]:
    saved_files = emit_run_artifacts(
        report=report, out_dir=run_dir, filename_prefix="report", console=console
    )
    _event(console, "PASS", "Run completed successfully!", emoji="✅")
    _event(console, "DATA", f"Report: {saved_files['json']}", emoji="📄")
    if run_config.event_path:
        _event(console, "DATA", f"Events: {run_config.event_path}", emoji="📝")
    return saved_files


__all__ = ["emit_run_artifacts", "postprocess_and_summarize"]
