"""Report layout exposes comparisons and thresholds without changing assurance."""

import re
from dataclasses import replace
from html import escape

import pytest

from invarlock.report_presentation import (
    IntervalView,
    MetricView,
    ReportView,
    _comparison_context,
    _detail_content,
    _interval,
    render_html,
)


def view(**kwargs):
    return ReportView(
        title="Report",
        family="Captured",
        decision="pass",
        summary="A result",
        metrics=(),
        assurance=(),
        **kwargs,
    )


def test_comparison_aligns_differences_and_keeps_shared_facts_once():
    report = view(
        subjects=(
            ("Baseline", "Observed model: model-a · Deployment: a"),
            ("Subject", "Observed model: model-b · Deployment: b"),
        ),
        context=(
            ("Baseline deployment", "a"),
            ("Subject deployment", "b"),
            ("Baseline observed model", "model-a"),
            ("Subject observed model", "model-b"),
            ("Baseline provider", "provider"),
            ("Subject provider", "provider"),
        ),
    )
    html = _comparison_context(report)
    assert 'class="comparison-table"' in html
    assert html.count('class="different"') == 2
    assert html.index("Observed model") < html.index("Deployment")
    assert "Subject identity" not in html
    assert "<dt>Provider</dt><dd>provider</dd>" in html
    assert "Additional recorded context (1)" in html
    assert report.subjects[0][1].startswith("Observed model:")


def test_comparison_preserves_incomplete_repeated_and_unknown_fields():
    report = view(
        subjects=(("Baseline", "model-a"), ("Subject", "model-b")),
        context=(
            ("Baseline revision", "first"),
            ("Baseline revision", "second"),
            ("Unknown <field>", "<script>"),
        ),
    )
    html = _comparison_context(report)
    assert "model-a" in html and "model-b" in html
    assert "first" in html and "second" in html
    assert "Unknown &lt;field&gt;" in html and "&lt;script&gt;" in html
    assert "<script>" not in html


def test_native_candidate_alias_aligns_identity_and_context_without_mutation():
    report = view(
        subjects=(
            ("Baseline", "baseline-artifact"),
            ("Candidate", "candidate-artifact"),
        ),
        context=(("Baseline provider", "cpu"), ("Candidate provider", "mps")),
    )
    html = _comparison_context(report)
    assert 'class="comparison-table"' in html
    assert re.search(
        r'<td data-side="Baseline">.*?baseline-artifact</td><td data-side="Subject">.*?candidate-artifact</td>',
        html,
    )
    assert re.search(
        r'<td data-side="Baseline">.*?cpu</td><td data-side="Subject">.*?mps</td>', html
    )
    assert report.subjects[1][0] == "Candidate"
    assert report.context[1][0] == "Candidate provider"


@pytest.mark.parametrize("subject_label", ["Subject artifact", "Candidate artifact"])
def test_subjects_with_matching_field_suffixes_align_and_keep_the_field_name(
    subject_label,
):
    report = view(
        subjects=(
            ("Baseline artifact", "baseline-artifact"),
            (subject_label, "subject-artifact"),
        ),
    )
    html = _comparison_context(report)
    assert 'class="comparison-table"' in html
    assert (
        '<th scope="row">Artifact<span class="different-label">Differs</span></th>'
        in html
    )
    assert re.search(
        r'<td data-side="Baseline">.*?baseline-artifact</td><td data-side="Subject">.*?subject-artifact</td>',
        html,
    )
    assert report.subjects[1][0] == subject_label


@pytest.mark.parametrize(
    "subjects",
    [
        (("Baseline artifact", "baseline"), ("Candidate service", "subject")),
        (("Baseline artifact", "baseline"), ("Reference artifact", "reference")),
        (("Baseline artifact", "first"), ("Baseline artifact", "second")),
        (
            ("Baseline artifact", "baseline"),
            ("Subject artifact", "subject"),
            ("Candidate artifact", "candidate"),
        ),
    ],
)
def test_ambiguous_or_unknown_suffixed_subjects_keep_original_labels(subjects):
    html = _comparison_context(view(subjects=subjects))
    assert 'class="comparison-table"' not in html
    for label, value in subjects:
        assert f"<dt>{label}</dt><dd>{value}</dd>" in html


@pytest.mark.parametrize(
    "extra_side", ["Reference", "Baseline", "Subject", "Candidate"]
)
def test_represented_pair_never_hides_additional_or_repeated_subjects(extra_side):
    report = view(
        subjects=(
            (extra_side, "additional-identity"),
            ("Baseline", "Observed model: model-a · Deployment: a"),
            ("Subject", "Observed model: model-b · Deployment: b"),
        ),
        context=(
            ("Baseline deployment", "a"),
            ("Subject deployment", "b"),
            ("Baseline observed model", "model-a"),
            ("Subject observed model", "model-b"),
        ),
    )
    html = _comparison_context(report)
    assert "additional-identity" in html
    assert "Observed model: model-a · Deployment: a" in html
    assert "Observed model: model-b · Deployment: b" in html


@pytest.mark.parametrize("last_label", ["Subject revision", "Candidate revision"])
def test_ambiguous_context_aliases_keep_all_values_under_their_original_labels(
    last_label,
):
    report = view(
        context=(
            ("Baseline revision", "baseline-revision"),
            ("Subject revision", "first-revision"),
            (last_label, "second-revision"),
        )
    )
    html = _comparison_context(report)
    assert 'class="comparison-table"' not in html
    assert "<dt>Baseline revision</dt><dd>baseline-revision</dd>" in html
    assert "<dt>Subject revision</dt><dd>first-revision</dd>" in html
    assert f"<dt>{last_label}</dt><dd>second-revision</dd>" in html


def test_equal_shortened_provider_labels_do_not_claim_equal_recorded_settings():
    from invarlock.record_reporting import _captured_context
    from tests.reporting.test_hosted_service_context import runs

    baseline, subject, _ = runs()
    baseline["service_identity"]["provider"] = "p" * 256 + "-baseline"
    subject["service_identity"]["provider"] = "p" * 256 + "-subject"
    subjects, context, changes, _ = _captured_context(
        {"baseline": baseline, "subject": subject}
    )
    html = _comparison_context(
        view(subjects=subjects, context=context, changes=changes)
    )
    assert "Matching displayed fields" in html
    assert "Matching previews do not establish equality" in html
    assert "<dt>Provider</dt>" in html
    assert "Shared settings" not in html
    assert "Declared service fields differ: provider." in html


@pytest.mark.parametrize("direction", [None, "minimum", "maximum", "unknown"])
@pytest.mark.parametrize("neutral", [None, 0.0, 1.0])
def test_interval_only_shades_an_explicit_change_requirement(direction, neutral):
    interval = IntervalView(-1, 3, 1, -2, "95% interval", "pp", direction, neutral)
    html = _interval(interval)
    assert ('class="allowed"' in html) == (direction in {"minimum", "maximum"})
    assert 'class="axis-labels"' in html and 'class="tick"' in html
    assert ('class="neutral"' in html) == (neutral is not None)
    assert "Policy threshold -2 pp" in html
    assert "acceptance" not in html
    if direction == "minimum":
        assert "≥ -2 pp" in html
    elif direction == "maximum":
        assert "≤ -2 pp" in html


def test_zero_width_interval_and_missing_threshold_stay_readable():
    html = _interval(IntervalView(1, 1, 1, None, "Interval <x>", "ratio"))
    assert "Interval &lt;x&gt;" in html
    assert html.count('class="tick"') == 1
    assert 'class="threshold"' not in html and 'class="allowed"' not in html
    assert 'class="legend"' not in html


def test_preview_text_is_escaped_without_parsing_elision_or_reencoding():
    data = {
        "preview_only": True,
        "text": '{\n  "field": <depth limit>\n}',
        "note": "<script>note</script>",
        "limits": {"depth": 6},
    }
    html = _detail_content(data)
    assert "&lt;depth limit&gt;" in html and "&lt;script&gt;" in html
    assert "\\&quot;" not in html and "<script>" not in html
    assert '"field"' in data["text"]
    assert "Preview limits" in html
    assert "Bounded preview" in _detail_content(
        {"preview_only": True, "text": "limited"}
    )
    assert "&quot;field&quot;" in _detail_content({"field": "value"})


def test_empty_checks_and_single_result_do_not_add_navigation_or_redundant_pills():
    from invarlock.report_presentation import CheckView

    metric = MetricView(
        "metric",
        "overall",
        "pass",
        "5%",
        "6%",
        "+1 pp",
        "400",
        "Result",
        checks=(CheckView("Count", "400", "400", True),),
    )
    html = render_html(replace(view(), metrics=(metric,)))
    assert 'class="check-detail"' not in html
    assert "Original decision:" not in html and "metric / scope result" not in html
    assert 'class="metric-navigation"' not in html
    assert 'class="checks-table"' in html
    assert "body{margin:0;overflow-wrap:anywhere" not in html


def test_nested_policy_previews_have_a_readable_escaped_disclosure():
    preview = {"preview_only": True, "text": "<img src=x> <node limit>"}
    report = view(
        details=(
            (
                "Policy",
                {
                    "metrics": [
                        None,
                        {"name": "<b>", "configuration": preview},
                        {"name": "ordinary", "configuration": {}},
                    ]
                },
            ),
        )
    )
    html = render_html(report)
    assert "&lt;b&gt; configuration preview" in html
    assert "<img" not in html and "&lt;node limit&gt;" in html
    assert "ordinary configuration preview" not in html


def test_summary_emphasis_is_escaped_and_does_not_split_other_numbers():
    from invarlock.report_presentation import _summary_html

    metric = MetricView("<img>", "overall", "pass", "5%", "6%", "+1 pp", "400", "")
    report = replace(
        view(), metrics=(metric,), summary="<img>: 5% to 6%; 95% interval; +1 pp."
    )
    rendered = _summary_html(report)
    assert "<strong>&lt;img&gt;</strong>" in rendered
    assert "<strong>5%</strong>" in rendered
    assert "95% interval" in rendered
    assert "9<strong>5%" not in rendered
    assert "<img>" not in rendered
    assert (
        _summary_html(replace(report, metrics=()))
        == "&lt;img&gt;: 5% to 6%; 95% interval; +1 pp."
    )
    empty = replace(metric, name="", baseline="", candidate="", change="")
    assert (
        _summary_html(replace(report, metrics=(empty,)))
        == "&lt;img&gt;: 5% to 6%; 95% interval; +1 pp."
    )


@pytest.mark.parametrize(
    "bounds",
    [
        (-1.444, 3.526, 1, -2),
        (1.085, 1.098, 1.091, 1.05),
        (0, 0, 0, None),
        (-1e308, 1e308, 0, 1e307),
        (1e-320, 2e-320, 1.5e-320, None),
    ],
)
def test_chart_annotations_remain_finite_and_show_the_actual_bounds(bounds):
    from invarlock.report_presentation import number

    lower, upper, estimate, threshold = bounds
    chart = _interval(
        IntervalView(
            lower, upper, estimate, threshold, "<interval>", "units", "minimum", 0
        )
    )
    assert "&lt;interval&gt;" in chart
    assert 'class="chart-annotations"' in chart
    assert 'class="estimate-key"' in chart
    assert number(lower) in chart and number(upper) in chart
    assert not re.search(r'(?:left|x1|x2|width|cx)[=:]"?(?:nan|inf|-inf)', chart)
    if threshold is not None:
        assert "Limit ≥ " + number(threshold) in chart


def test_metric_sublabels_and_count_label_are_escaped_and_assurance_is_not_duplicated():
    metric = MetricView(
        "quality",
        "overall",
        "pass",
        "5%",
        "6%",
        "+1 pp",
        "400",
        "",
        baseline_detail="<script>",
        candidate_detail="24 matched",
        count_detail="0 missing",
        count_label="<pairs>",
    )
    report = replace(
        view(), metrics=(metric,), assurance=(("Signing", "Not authorized"),)
    )
    html = render_html(report)
    assert "<small>&lt;script&gt;</small>" in html
    assert "<small>24 matched</small>" in html
    assert "<dt>&lt;pairs&gt;</dt>" in html
    assert html.count("Not authorized") == 1
    assert html.index("Not authorized") < html.index("Results and requirements")


def test_neutral_reference_is_not_given_a_policy_limit_key_without_policy():
    chart = _interval(IntervalView(1, 2, 1.5, None, "Range", "ratio", neutral=1))
    assert 'class="neutral-key"' in chart
    assert 'class="threshold-key"' not in chart
    assert "No change: 1 ratio." in chart
    assert "Limit" not in chart


def test_regular_ticks_use_display_units_and_summary_emphasizes_sentence_end_value():
    from invarlock.report_presentation import _summary_html

    chart = _interval(
        IntervalView(-1.444, 3.526, 1, -2, "Paired interval", "pp", "minimum", 0)
    )
    assert re.search(r'class="axis-labels".*?>-2</span>.*?>0</span>.*?>2</span>', chart)
    assert ">1.763</span>" not in chart
    metric = MetricView("quality", "overall", "pass", "5%", "6%", "+1 pp", "400", "")
    assert "<strong>+1 pp</strong>." in _summary_html(
        replace(view(), metrics=(metric,), summary="Change +1 pp.")
    )


def test_grouped_comparison_is_shared_by_html_and_markdown_without_loss():
    from invarlock.report_presentation import _comparison_view, render_markdown

    report = view(
        subjects=(("Baseline artifact", "first"), ("Candidate artifact", "second")),
        context=(
            ("Baseline HTTP roles", "system,user"),
            ("Subject HTTP roles", "system,user"),
            ("Baseline revision", "a"),
            ("Baseline revision", "b"),
        ),
    )
    grouped = _comparison_view(report)
    assert grouped.rows == (("Artifact", "first", "second", True),)
    assert grouped.matching == (("HTTP roles", "system,user"),)
    assert grouped.additional == (
        ("Baseline revision", "a"),
        ("Baseline revision", "b"),
    )
    for rendered in (render_html(report), render_markdown(report)):
        assert "Matching displayed fields" in rendered
        assert "Matching previews do not establish equality" in rendered
        assert "HTTP roles" in rendered
        assert "Differs" in rendered
        assert rendered.count("Baseline revision") == 2
        assert "same displayed text" not in rendered
    assert "### Artifact — Differs" in render_markdown(report)
    assert "- **Baseline:** first\n- **Subject:** second" in render_markdown(report)


@pytest.mark.parametrize(
    "decision,opened",
    [
        ("pass", False),
        ("fail", True),
        ("regression", True),
        ("insufficient_evidence", True),
    ],
)
def test_adverse_metric_overview_starts_open(decision, opened):
    first = MetricView("quality", "overall", "pass", "1", "1", "0", "20", "")
    second = replace(first, name="latency", decision=decision)
    html = render_html(replace(view(), metrics=(first, second)))
    assert ('<details class="overview-disclosure" open>' in html) is opened
    assert '<details class="overview-disclosure">' in html or opened


def test_checks_have_mobile_labels_and_dark_theme_is_screen_only():
    from invarlock.report_presentation import _CSS, CheckView

    metric = MetricView(
        "quality",
        "overall",
        "fail",
        "1",
        "0",
        "-1",
        "20",
        "",
        checks=(CheckView("Accuracy", "0%", "≥ 95%", False),),
    )
    html = render_html(replace(view(), metrics=(metric,)))
    for label in ("Observed", "Required", "Result"):
        assert f'data-label="{label}"' in html
        assert f'aria-hidden="true">{label}</span>' in html
    assert "min-width:540px" not in _CSS
    assert "@media screen and (prefers-color-scheme:dark)" in _CSS
    assert _CSS.index("prefers-color-scheme:dark") < _CSS.index("@media print")
    assert "default-src 'none'" in html


def test_normal_axis_domain_ends_have_ticks_and_geometry_agrees():
    chart = _interval(
        IntervalView(-1.444, 3.526, 1, -2, "Interval", "pp", "minimum", 0)
    )
    assert ">-4</span>" in chart and ">6</span>" in chart
    assert 'x1="30.00" x2="30.00"' in chart
    assert 'x1="610.00" x2="610.00"' in chart
    # -2 is one fifth of the way from -4 to 6, including the 30px margin.
    assert 'class="threshold" x1="146.00" x2="146.00"' in chart


@pytest.mark.parametrize("lower", [1e-306, 1.0, 1e148, 4.571237497287615e307])
def test_rounded_domain_cannot_collapse_or_exclude_adjacent_bounds(lower):
    import math

    upper = math.nextafter(lower, math.inf)
    chart = _interval(
        IntervalView(lower, upper, lower, upper, "Interval", "score", "minimum")
    )
    positions = [float(value) for value in re.findall(r'(?:x1|x2|cx)="([^"]+)"', chart)]
    assert positions
    assert all(30 <= value <= 610 for value in positions)


@pytest.mark.parametrize(
    ("label", "display_label"),
    [
        ("Paired 95% confidence interval", "Paired 95% CI"),
        (
            "95% finite-schedule resampling interval",
            "95% finite-schedule resampling interval",
        ),
        (
            "Paired independent-unit effect interval",
            "Paired independent-unit effect interval",
        ),
        ("<custom interval>", "&lt;custom interval&gt;"),
    ],
)
def test_change_tile_shows_bound_interval_without_relabelling_method(
    label, display_label
):
    metric = MetricView(
        "Metric",
        "Overall",
        "pass",
        "5%",
        "6%",
        "+1 pp",
        "400",
        "Comparison",
        interval=IntervalView(-1.444, 3.526, 1, -2, label, "pp", "minimum", 0),
    )
    html = render_html(replace(view(), metrics=(metric,)))
    assert (
        f"<dt>Change</dt><dd>+1 pp<small>{display_label}: -1.444 to +3.526 pp</small></dd>"
        in html
    )
    chart_summary = f"{escape(label)}: -1.444 to +3.526 pp"
    assert f"{escape(label)}</span>" in html
    assert f'aria-label="{chart_summary}. Estimate +1 pp.' in html
    html = render_html(replace(view(), metrics=(replace(metric, interval=None),)))
    assert "<dt>Change</dt><dd>+1 pp</dd>" in html


def test_chart_names_retained_method_while_tile_stays_compact():
    interval = IntervalView(
        -1.444,
        3.526,
        1,
        -2,
        "Paired 95% confidence interval",
        "pp",
        "minimum",
        0,
        "Newcombe hybrid score",
    )
    chart = _interval(interval)
    assert (
        "Paired 95% confidence interval (Newcombe hybrid score): -1.444 to +3.526 pp"
        in chart
    )
    assert "&lt;method&gt;" in _interval(replace(interval, method="<method>"))
    caption = chart.split("<figcaption>", 1)[1].split("</figcaption>", 1)[0]
    assert "Paired 95% confidence interval (Newcombe hybrid score)</span>" in caption
    assert "Estimate</span>" in caption
    assert "Policy threshold</span>" in caption
    assert "No change</span>" in caption
    assert "Allowed change region</span>" in caption
    assert all(value not in caption for value in ("-1.444", "+3.526", "+1 pp", "-2 pp"))


def test_ratio_change_tile_does_not_present_ratio_bounds_as_signed_effects():
    metric = MetricView(
        "NLL",
        "Overall",
        "pass",
        "1",
        "1.01",
        "1.01×",
        "400",
        "Comparison",
        interval=IntervalView(
            0.98,
            1.03,
            1.01,
            1.1,
            "95% finite-schedule resampling interval",
            "ratio",
            "maximum",
            1,
        ),
    )
    html = render_html(replace(view(), metrics=(metric,)))
    assert (
        "<small>95% finite-schedule resampling interval: 0.98 to 1.03 ratio</small>"
        in html
    )
