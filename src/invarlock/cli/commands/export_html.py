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
from rich.console import Console

console = Console()


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

    if out_path.exists() and not force:
        console.print(
            f"[red]❌ Output file already exists. Use --force to overwrite: {out_path}[/red]"
        )
        raise typer.Exit(1)

    try:
        payload = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]❌ Failed to read input JSON: {exc}[/red]")
        raise typer.Exit(1) from exc

    try:
        from invarlock.reporting.html import render_report_html

        html = render_report_html(payload)
    except ValueError as exc:
        # Evaluation report validation failed upstream
        console.print(f"[red]❌ Evaluation report validation failed: {exc}[/red]")
        raise typer.Exit(2) from exc
    except Exception as exc:
        console.print(f"[red]❌ Failed to render HTML: {exc}[/red]")
        raise typer.Exit(1) from exc

    if not embed_css:
        # Strip <style>...</style> from the head to leave it bare
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
    except Exception as exc:
        console.print(f"[red]❌ Failed to write output file: {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"✅ Exported evaluation report HTML → {out_path}")


__all__ = ["export_html_command"]
