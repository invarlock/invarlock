"""
InvarLock HTML Export
=================

 Thin wrapper over the HTML evaluation report renderer to make exporting
 discoverable and scriptable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import typer

from invarlock.cli import output as cli_output

console = cli_output.make_console()
_JSON_INPUT_ERRORS = (OSError, UnicodeDecodeError, json.JSONDecodeError)
_HTML_RENDER_ERRORS = (AttributeError, ImportError, RuntimeError, TypeError)
_HTML_OUTPUT_ERRORS = (OSError, UnicodeEncodeError)


def _load_html_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_html_payload(payload: dict[str, object]) -> str:
    from invarlock.reporting.html import render_report_html

    return render_report_html(payload)


def _write_html_payload(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def export_html_command(
    input: str,
    output: str,
    embed_css: bool = True,
    force: bool = False,
) -> None:
    """Render an evaluation report JSON to HTML.

    Exit codes:
    - 0: success
    - 1: generic failure (IO or overwrite refusal)
    - 2: validation failure (invalid evaluation report schema)
    """
    in_path = Path(str(input))
    out_path = Path(str(output))
    emit = cli_output.make_command_event_emitter(console)

    if out_path.exists() and not force:
        emit("FAIL", "Output file already exists")
        cli_output.print_command_detail(
            console, f"Use --force to overwrite: {out_path}"
        )
        raise typer.Exit(1)

    try:
        payload = _load_html_payload(in_path)
    except _JSON_INPUT_ERRORS as exc:
        emit("FAIL", f"Failed to read input JSON: {exc}")
        raise typer.Exit(1) from exc

    try:
        html = _render_html_payload(payload)
    except ValueError as exc:
        # Evaluation report validation failed upstream
        emit("FAIL", f"Evaluation report validation failed: {exc}")
        raise typer.Exit(2) from exc
    except _HTML_RENDER_ERRORS as exc:
        emit("FAIL", f"Failed to render HTML: {exc}")
        raise typer.Exit(1) from exc

    if not embed_css:
        # Strip <style>...</style> from the head to leave it bare
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )

    try:
        _write_html_payload(out_path, html)
    except _HTML_OUTPUT_ERRORS as exc:
        emit("FAIL", f"Failed to write output file: {exc}")
        raise typer.Exit(1) from exc

    emit("PASS", "Exported evaluation report HTML")
    cli_output.print_command_detail(console, f"Input: {in_path}")
    cli_output.print_command_detail(console, f"Output: {out_path}")


__all__ = ["export_html_command"]
