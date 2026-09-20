"""Display labels never replace the metric and scope identities they describe."""

from copy import deepcopy
from dataclasses import replace
from xml.etree.ElementTree import fromstring

import pytest

from invarlock.record_reporting import _captured_summary, render_junit
from invarlock.report_presentation import (
    MetricView,
    ReportView,
    display_label,
    render_html,
    render_markdown,
)
from tests.reporting.test_captured_summary import binary_comparison


@pytest.mark.parametrize(
    ("raw", "display"),
    [
        ("grounded_qa-judge-quality", "Grounded QA judge quality"),
        ("west_coast", "West coast"),
        ("nll_per_utf-8_byte", "NLL per UTF-8 byte"),
        ("api_http_f1", "API HTTP F1"),
        ("UTF-8", "UTF-8"),
        ("nll_per_utf_8_byte", "NLL per UTF-8 byte"),
        ("utf_quality", "Utf quality"),
        ("f1", "F1"),
        ("nll", "NLL"),
        ("Grounded QA: judge quality", "Grounded QA: judge quality"),
        ("F1 / API HTTP", "F1 / API HTTP"),
        ("All paired records", "All paired records"),
        ("org/model-7b", "org/model-7b"),
        ("pass_at_1", "Pass at 1"),
        ("quality_v2", "Quality v2"),
        ("cohort_2", "Cohort 2"),
        ("sha256:abcdef", "sha256:abcdef"),
        ("<img src=x onerror=alert(1)>", "<img src=x onerror=alert(1)>"),
    ],
)
def test_conservative_metric_labels_preserve_prose_acronyms_and_identifiers(
    raw, display
):
    assert display_label(raw) == display


def _view(*names):
    return ReportView(
        title="Comparison",
        family="Captured records",
        decision="pass",
        summary="Recorded result.",
        metrics=tuple(
            MetricView(name, "west_coast", "pass", "80%", "90%", "+10 pp", "20", "")
            for name in names
        ),
        assurance=(),
        context=(("Requested model", "org/model-7b"), ("Run", "baseline_run-v1")),
        technical={"source": "external_evaluator", "rubric": "exact_match-only"},
    )


def test_single_metric_labels_and_raw_identifiers_are_available_in_both_formats():
    view = _view("grounded_qa-judge-quality")
    original = deepcopy(view)
    for output in (render_html(view), render_markdown(view)):
        assert "Grounded QA judge quality" in output
        assert "West coast" in output
        assert "Recorded metric and scope identifiers" in output
        assert "grounded_qa-judge-quality" in output
        assert "west_coast" in output
        assert "org/model-7b" in output
        assert "baseline_run-v1" in output
    assert view == original
    assert view.metrics[0].name == "grounded_qa-judge-quality"
    assert view.metrics[0].scope == "west_coast"


def test_display_collisions_keep_distinct_groups_and_result_links():
    view = _view("grounded_qa", "grounded-qa")
    html = render_html(view)
    for index in (1, 2):
        assert f'id="metric-group-{index}"' in html
        assert f'href="#metric-group-{index}"' in html
        assert f'id="metric-result-{index}"' in html
    assert html.count('class="metric-group"') == 2
    for output in (html, render_markdown(view)):
        assert "grounded_qa" in output and "grounded-qa" in output
        assert "Grounded QA" in output


def test_metric_labels_and_identifier_details_remain_escaped():
    raw = '<script>alert("x")</script>_quality'
    view = _view(raw)
    html = render_html(view)
    assert "<script>alert(" not in html
    assert "&lt;script&gt;" in html
    markdown = render_markdown(view)
    assert raw not in markdown.split("```json")[0]
    assert "&lt;script&gt;" in markdown


def test_captured_summaries_use_display_labels_but_junit_keeps_raw_names():
    _, comparison, _ = binary_comparison()
    comparison["metrics"][0]["name"] = "grounded_qa-judge-quality"
    comparison["metrics"][0]["slice"] = "west_coast"
    original = deepcopy(comparison)
    view = _view("grounded_qa-judge-quality")
    assert "subject's Grounded QA judge quality" in _captured_summary(
        view.metrics, comparison
    )
    assert "within the West coast slice" in _captured_summary(view.metrics, comparison)
    adverse = replace(view.metrics[0], decision="regression")
    assert _captured_summary((adverse, view.metrics[0])).startswith(
        "Grounded QA judge quality (West coast): No explanation was supplied"
    )
    case = fromstring(render_junit(comparison)).find("testcase")
    assert case.get("name") == "grounded_qa-judge-quality"
    assert case.get("classname") == "west_coast"
    assert comparison == original


def test_long_invalid_identifier_is_rejected_without_excessive_backtracking():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from invarlock.report_presentation import display_label; "
            "label = 'f1_' * 2000 + '!'; assert display_label(label) == label",
        ],
        check=True,
        capture_output=True,
        timeout=5,
    )
