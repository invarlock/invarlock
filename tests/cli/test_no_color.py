from __future__ import annotations

from io import StringIO


def test_no_color_avoids_ansi(monkeypatch) -> None:
    from invarlock.cli.output import make_console

    buf = StringIO()
    monkeypatch.delenv("NO_COLOR", raising=False)
    console_color = make_console(file=buf, force_terminal=True)
    console_color.print("hello", style="red")
    colored = buf.getvalue()
    assert "\x1b[" in colored

    buf = StringIO()
    monkeypatch.setenv("NO_COLOR", "1")
    console_plain = make_console(file=buf, force_terminal=True)
    console_plain.print("hello", style="red")
    plain = buf.getvalue()
    assert "\x1b[" not in plain
    assert "hello" in plain
