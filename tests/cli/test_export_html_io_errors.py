from __future__ import annotations

from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.export_html import export_html_command


def test_export_html_missing_input_exit1(tmp_path: Path):
    out = tmp_path / "out.html"
    with pytest.raises(typer.Exit) as exc:
        export_html_command(
            input=str(tmp_path / "missing.json"),
            output=str(out),
            embed_css=True,
            force=True,
        )
    assert exc.value.exit_code == 1


def test_export_html_write_error_exit1(monkeypatch, tmp_path: Path):
    # Valid minimal input
    inp = tmp_path / "in.json"
    inp.write_text("{}", encoding="utf-8")
    # Ensure rendering succeeds then force write failure
    import invarlock.reporting.html as html_mod

    monkeypatch.setattr(html_mod, "render_certificate_html", lambda payload: "<html/>")

    # Force write failure
    def _boom(*a, **k):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _boom)
    with pytest.raises(typer.Exit) as exc:
        export_html_command(
            input=str(inp),
            output=str(tmp_path / "out.html"),
            embed_css=True,
            force=True,
        )
    assert exc.value.exit_code == 1
