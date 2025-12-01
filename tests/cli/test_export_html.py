from __future__ import annotations

from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.export_html import export_html_command


def test_export_html_refuses_overwrite(tmp_path: Path):
    # Create a dummy existing output file
    out = tmp_path / "out.html"
    out.write_text("x", encoding="utf-8")
    # Input path need not exist because overwrite check happens first
    with pytest.raises(typer.Exit) as ei:
        export_html_command(
            input=str(tmp_path / "in.json"),
            output=str(out),
            embed_css=True,
            force=False,
        )
    assert ei.value.exit_code == 1
