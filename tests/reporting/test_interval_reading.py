"""Interval guidance preserves supplied decisions and method-specific evidence."""

from dataclasses import asdict

import pytest

from invarlock.report_presentation import (
    IntervalView,
    MetricView,
    ReportView,
    _interval_reading,
    render_html,
    render_markdown,
)


def view(interval):
    return ReportView(
        title="Comparison",
        family="Recorded comparison",
        decision="fail",
        summary="The recorded policy was not met.",
        assurance=(),
        metrics=(
            MetricView(
                name="Accuracy",
                scope="Recorded pairs",
                decision="fail",
                baseline="73.44%",
                candidate="72.66%",
                change="-0.7812 pp",
                count="128",
                explanation="Lower-bound requirement not met.",
                interval=interval,
            ),
        ),
    )


@pytest.mark.parametrize(
    "direction,endpoint,operator",
    [
        ("minimum", "lower bound (left endpoint)", "at least"),
        ("maximum", "upper bound (right endpoint)", "at most"),
    ],
)
def test_policy_explanation_uses_the_declared_endpoint(direction, endpoint, operator):
    interval = IntervalView(
        -4.10336,
        2.4968,
        -0.78125,
        -2,
        "Paired 95% confidence interval",
        "pp",
        direction,
        0,
        "Newcombe hybrid score",
        ("Paired outcomes, 128 records.",),
    )
    report = view(interval)
    before = asdict(report)
    rule = dict(_interval_reading(interval))["Policy rule"]
    assert endpoint in rule and operator in rule
    for render in (render_html, render_markdown):
        output = render(report)
        assert "How to read this comparison" in output
        assert "How this interval was calculated" in output
        assert "Paired outcomes, 128 records." in output
        assert "Newcombe hybrid score" in output
        assert "Policy not met" in output
    assert asdict(report) == before


@pytest.mark.parametrize("threshold,direction", [(None, "minimum"), (1, None)])
def test_missing_threshold_rule_is_not_inferred(threshold, direction):
    interval = IntervalView(
        0.9, 1.2, 1.05, threshold, "Schedule interval", "ratio", direction
    )
    explanation = dict(_interval_reading(interval))
    assert "Policy rule" not in explanation
    assert "No change" not in explanation
    for render in (render_html, render_markdown):
        assert "How this interval was calculated" not in render(view(interval))


def test_unavailable_interval_has_no_invented_explanation():
    for render in (render_html, render_markdown):
        output = render(view(None))
        assert "How to read this comparison" not in output
        assert "How this interval was calculated" not in output


def test_calculation_details_cannot_inject_markup_or_terminal_controls():
    interval = IntervalView(
        0,
        1,
        0.5,
        None,
        "Interval",
        "score",
        basis=('<script>alert("x")</script>\x1b[2J',),
    )
    for render in (render_html, render_markdown):
        output = render(view(interval))
        assert "<script>" not in output
        assert "&lt;script&gt;" in output
        assert "\x1b" not in output
