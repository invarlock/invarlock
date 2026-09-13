"""Report layout exposes comparisons and thresholds without changing assurance."""

from dataclasses import replace

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
    assert "<dt>Provider (same displayed text)</dt><dd>provider</dd>" in html
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
    assert "<td>baseline-artifact</td><td>candidate-artifact</td>" in html
    assert "<td>cpu</td><td>mps</td>" in html
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
    assert '<th scope="row">Artifact</th>' in html
    assert "<td>baseline-artifact</td><td>subject-artifact</td>" in html
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
    assert "Provider (same displayed text)" in html
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
