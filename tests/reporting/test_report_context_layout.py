"""Comparison context remains visible and safe across shared report formats."""

from itertools import product

import pytest

from invarlock.report_presentation import ReportView, render_html, render_markdown


@pytest.mark.parametrize(
    "subjects,context,changes", list(product((False, True), repeat=3))
)
def test_context_layout_is_optional_and_precedes_results(subjects, context, changes):
    view = ReportView(
        title="Comparison",
        family="Captured results",
        decision="fail",
        summary="A recorded regression.",
        metrics=(),
        assurance=(("Replay", "Not performed"),),
        subjects=(("Baseline", "model-a"),) if subjects else (),
        context=(("Task", "routing"),) if context else (),
        changes=("System instruction added.",) if changes else (),
    )
    for render in (render_html, render_markdown):
        output = render(view)
        assert ("What was compared" in output) == any((subjects, context, changes))
        assert ("Recorded changes" in output) == changes
        if subjects:
            assert "model-a" in output
        if context:
            assert "routing" in output
        if changes:
            assert "System instruction added." in output
        if any((subjects, context, changes)):
            assert output.index("What was compared") < output.index("What was checked")


def test_untrusted_context_is_escaped_in_both_formats():
    unsafe = '<script>alert("x")</script> [click](https://example.org)'
    view = ReportView(
        title="Comparison",
        family="Native",
        decision="pass",
        summary="Result",
        metrics=(),
        assurance=(),
        subjects=((unsafe, unsafe),),
        context=((unsafe, unsafe),),
        changes=(unsafe,),
    )
    for render in (render_html, render_markdown):
        output = render(view)
        assert "<script>" not in output
        assert "&lt;script&gt;" in output
    assert "[click](https://example.org)" not in render_markdown(view)


@pytest.mark.parametrize("control", ["\x1b[2J\x1b[H", "\x9b2J", "\x00", "\x07", "\r"])
def test_context_cannot_emit_terminal_controls(control):
    from io import StringIO

    from rich.console import Console
    from rich.markdown import Markdown

    view = ReportView(
        title="Comparison",
        family="Captured",
        decision="fail",
        summary="Result",
        metrics=(),
        assurance=(),
        context=(("Model", "safe" + control + "FAKE_PASS"),),
    )
    markdown = render_markdown(view)
    stream = StringIO()
    Console(file=stream, force_terminal=False).print(Markdown(markdown))
    for output in (render_html(view), markdown, stream.getvalue()):
        assert control not in output
    assert "FAKE_PASS" in stream.getvalue()


def test_expanded_json_cannot_emit_terminal_controls():
    control = "\x9b2J"
    view = ReportView(
        title="Comparison",
        family="Captured",
        decision="fail",
        summary="Result",
        metrics=(),
        assurance=(),
        technical={"metric": "safe" + control + "FAKE_PASS"},
    )
    markdown = render_markdown(view, include_details=True)
    assert control not in markdown
    assert "safe\\u009b2JFAKE_PASS" in markdown
