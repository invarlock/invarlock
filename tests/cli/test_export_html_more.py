from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.models import OptionInfo

from invarlock.cli.commands.export_html import export_html_command


def test_export_html_success_and_strip_css(monkeypatch, tmp_path: Path):
    # Monkeypatch renderer to return HTML with a style tag
    import invarlock.reporting.html as html_mod

    html = "<html><head><style>.x{}</style></head><body>ok</body></html>"
    monkeypatch.setattr(html_mod, "render_certificate_html", lambda payload: html)

    inp = tmp_path / "in.json"
    out = tmp_path / "out.html"
    inp.write_text("{}", encoding="utf-8")

    # Strip CSS branch
    export_html_command(input=str(inp), output=str(out), embed_css=False, force=False)
    text = out.read_text(encoding="utf-8")
    assert "<style" not in text and "ok" in text


def test_export_html_validation_error_exit2(monkeypatch, tmp_path: Path):
    import invarlock.reporting.html as html_mod

    def _raise(_):  # type: ignore[no-untyped-def]
        raise ValueError("bad cert")

    monkeypatch.setattr(html_mod, "render_certificate_html", _raise)
    inp = tmp_path / "in.json"
    out = tmp_path / "out.html"
    inp.write_text("{}", encoding="utf-8")
    # Direct function call raises typer.Exit(2)
    import pytest

    with pytest.raises(typer.Exit) as exc:
        export_html_command(input=str(inp), output=str(out), embed_css=True, force=True)
    assert exc.value.exit_code == 2


def test_export_html_force_overwrite(monkeypatch, tmp_path: Path):
    import invarlock.reporting.html as html_mod

    monkeypatch.setattr(
        html_mod, "render_certificate_html", lambda payload: "<html>ok</html>"
    )
    inp = tmp_path / "in.json"
    out = tmp_path / "out.html"
    inp.write_text("{}", encoding="utf-8")
    out.write_text("old", encoding="utf-8")
    # Force overwrite allowed
    export_html_command(input=str(inp), output=str(out), embed_css=True, force=True)
    assert out.read_text(encoding="utf-8").strip() == "<html>ok</html>"


def test_export_html_coerces_optioninfo_defaults(monkeypatch, tmp_path: Path):
    import invarlock.reporting.html as html_mod

    monkeypatch.setattr(
        html_mod, "render_certificate_html", lambda payload: "<html>inline</html>"
    )
    inp = tmp_path / "in.json"
    out = tmp_path / "out.html"
    inp.write_text("{}", encoding="utf-8")

    inp_opt = OptionInfo()
    inp_opt.default = str(inp)
    out_opt = OptionInfo()
    out_opt.default = str(out)
    embed_opt = OptionInfo()
    embed_opt.default = False
    force_opt = OptionInfo()
    force_opt.default = True

    export_html_command(
        input=inp_opt, output=out_opt, embed_css=embed_opt, force=force_opt
    )
    assert out.exists()


def test_export_html_render_generic_error(monkeypatch, tmp_path: Path):
    import invarlock.reporting.html as html_mod

    def _raise(_payload):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(html_mod, "render_certificate_html", _raise)
    inp = tmp_path / "in.json"
    out = tmp_path / "out.html"
    inp.write_text("{}", encoding="utf-8")

    with pytest.raises(typer.Exit) as exc:
        export_html_command(input=str(inp), output=str(out), embed_css=True, force=True)
    assert exc.value.exit_code == 1
