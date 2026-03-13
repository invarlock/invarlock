from __future__ import annotations

from invarlock.cli import output as mod


class _RecordingConsole:
    def __init__(self, *, fail_with_kwargs: bool = False) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self._fail_with_kwargs = fail_with_kwargs

    def print(self, *args: object, **kwargs: object) -> None:
        if self._fail_with_kwargs and kwargs:
            self._fail_with_kwargs = False
            raise TypeError("kwargs unsupported")
        self.calls.append((args, kwargs))


def test_output_style_resolution_branches(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    audit = mod.OutputStyle(name="audit")
    friendly = mod.OutputStyle(name="friendly")
    assert audit.audit is True
    assert audit.emojis is False
    assert friendly.audit is False
    assert friendly.emojis is True

    assert mod.normalize_style(None) is None
    assert mod.normalize_style("   ") is None
    assert mod.normalize_style("friendly") == "friendly"
    assert mod.normalize_style("unknown") is None

    assert mod.resolve_style_name(None, "release") == "audit"
    assert mod.resolve_style_name(None, "dev") == "friendly"
    assert mod.resolve_style_name(" friendly ", "release") == "friendly"

    monkeypatch.setenv("NO_COLOR", "1")
    resolved = mod.resolve_output_style(
        style=None,
        profile="dev",
        progress=True,
        timing=True,
        no_color=False,
    )
    assert resolved.name == "friendly"
    assert resolved.progress is True
    assert resolved.timing is True
    assert resolved.color is False

    explicit = mod.resolve_output_style(
        style="audit",
        profile="dev",
        progress=False,
        timing=False,
        no_color=True,
    )
    assert explicit.name == "audit"
    assert explicit.color is False


def test_output_printing_and_style_assignment(monkeypatch) -> None:
    fallback_console = _RecordingConsole(fail_with_kwargs=True)
    mod._safe_console_print(fallback_console, "hello", style="green", markup=False)
    assert fallback_console.calls == [(("hello",), {})]

    recorder = _RecordingConsole()
    audit = mod.OutputStyle(name="audit", color=True)
    friendly = mod.OutputStyle(name="friendly", color=True)
    no_color = mod.OutputStyle(name="friendly", color=False)

    mod.print_event(recorder, "pass", "ok", style=audit)
    mod.print_event(recorder, "fail", "bad", style=audit)
    mod.print_event(recorder, "warn", "careful", style=audit)
    mod.print_event(recorder, "metric", "ratio=1.2", style=audit)
    mod.print_event(
        recorder,
        "",
        "uses default tag",
        style=friendly,
        emoji="✅",
        console_style="magenta",
    )
    mod.print_event(recorder, "info", "still plain", style=audit)
    mod.print_event(recorder, "info", "plain", style=no_color, emoji="🎯")

    lines = [call[0][0] for call in recorder.calls]
    kwargs = [call[1] for call in recorder.calls]
    assert lines[0] == "[PASS] ok"
    assert kwargs[0]["style"] == "green"
    assert kwargs[1]["style"] == "red"
    assert kwargs[2]["style"] == "yellow"
    assert kwargs[3]["style"] == "cyan"
    assert lines[4] == "✅ uses default tag"
    assert kwargs[4]["style"] == "magenta"
    assert lines[5] == "[INFO] still plain"
    assert kwargs[5]["style"] is None
    assert lines[6] == "🎯 plain"
    assert kwargs[6]["style"] is None

    assert mod.format_event_line("", "message", style=audit) == "[INFO] message"
    assert (
        mod.format_event_line("warn", "notice  ", style=friendly, emoji="⚠️")
        == "⚠️ notice"
    )


def test_output_timing_helpers_cover_progress_and_summary(monkeypatch) -> None:
    console = _RecordingConsole()
    style = mod.OutputStyle(name="audit", progress=True, timing=True, color=True)
    perf_values = iter([10.0, 12.25, 20.0, 20.5])
    monkeypatch.setattr(mod, "perf_counter", lambda: next(perf_values))

    timings: dict[str, float] = {}
    with mod.timed_step(
        console=console,
        style=style,
        timings=timings,
        key="phase",
        tag="pass",
        message="phase",
        emoji="✅",
    ):
        pass
    assert timings["phase"] == 2.25
    assert console.calls[-1][0][0] == "[PASS] phase done (2.25s)"

    no_progress = mod.OutputStyle(
        name="friendly", progress=False, timing=False, color=True
    )
    with mod.timed_step(
        console=console,
        style=no_progress,
        timings=None,
        key="silent",
        tag="info",
        message="skip",
    ):
        pass
    assert "silent" not in timings

    before_summary = len(console.calls)
    mod.print_timing_summary(
        console,
        {"phase": 2.25},
        style=mod.OutputStyle(name="audit", timing=False),
        order=[("Phase", "phase")],
    )
    assert len(console.calls) == before_summary

    mod.print_timing_summary(
        console,
        {"phase": 2.25},
        style=style,
        order=[("Phase", "phase"), ("Missing", "missing")],
        extra_lines=["  extra"],
    )
    rendered = [call[0][0] for call in console.calls[before_summary:]]
    assert rendered[0] == ""
    assert rendered[1] == "TIMING SUMMARY"
    assert rendered[2].startswith("  Phase")
    assert rendered[-1] == "  extra"
