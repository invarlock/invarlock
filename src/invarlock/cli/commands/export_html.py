"""
InvarLock HTML Export
=================

 Thin wrapper over the HTML certificate renderer to make exporting
 discoverable and scriptable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

console = Console()


def export_html_command(
    input: str = typer.Option(..., "--input", "-i", help="Path to certificate JSON"),
    output: str = typer.Option(..., "--output", "-o", help="Path to output HTML file"),
    embed_css: bool = typer.Option(
        True,
        "--embed-css/--no-embed-css",
        help="Inline a minimal static stylesheet (on by default)",
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite output file if it already exists"
    ),
) -> None:
    """Render a certificate JSON to HTML.

    Exit codes:
    - 0: success
    - 1: generic failure (IO or overwrite refusal)
    - 2: validation failure (invalid certificate schema)
    """
    # When called programmatically, Typer's Option defaults can be OptionInfo
    try:  # pragma: no cover - defensive, matches other commands' pattern
        from typer.models import OptionInfo as _TyperOptionInfo
    except Exception:  # pragma: no cover
        _TyperOptionInfo = ()  # type: ignore[assignment]

    def _coerce(value: Any) -> Any:
        if isinstance(value, _TyperOptionInfo):
            return value.default
        return value

    input = _coerce(input)
    output = _coerce(output)
    embed_css = bool(_coerce(embed_css))
    force = bool(_coerce(force))

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
        from invarlock.reporting.html import render_certificate_html

        html = render_certificate_html(payload)
    except ValueError as exc:
        # Certificate validation failed upstream
        console.print(f"[red]❌ Certificate validation failed: {exc}[/red]")
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

    console.print(f"✅ Exported certificate HTML → {out_path}")


__all__ = ["export_html_command"]
