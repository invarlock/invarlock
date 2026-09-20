"""Report titles and descriptions explain scoped decisions without inventing facts."""

from copy import deepcopy
from dataclasses import replace
from html import escape

import pytest

from invarlock.evaluation_records.templates import example_project
from invarlock.record_reporting import _captured_summary, _metric_views, _view
from invarlock.report_presentation import decision_label, render_html, render_markdown
from tests._evaluation_support import build_pack, pack_json


def captured(scenario):
    baseline, subject, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    metric = policy["metrics"][0]
    for run in (baseline, subject):
        for row in run["records"]:
            row["expected"] = row["output"] = "yes"
    if scenario == "count":
        metric["minimum_count"] = 50
    elif scenario == "width":
        metric["maximum_interval_width"] = 0.00001
    elif scenario == "floor":
        metric["subject_minimum"] = 0.99
        subject["records"][0]["output"] = "no"
    elif scenario == "ceiling":
        metric["subject_minimum"] = 0
        metric["subject_maximum"] = 0.9
    elif scenario == "missing":
        subject["records"][0]["output"] = None
        subject["records"][0]["error"] = "capture incomplete"
    elif scenario == "loss":
        for row in subject["records"]:
            row["output"] = "no"
    pack = build_pack(baseline, subject, policy)
    return pack, pack_json(pack, "report")


@pytest.mark.parametrize(
    "scenario,decision,cause",
    [
        ("pass", "pass", "uncertainty interval is within the required range"),
        ("count", "insufficient_evidence", "fewer paired records"),
        ("width", "insufficient_evidence", "interval is wider"),
        ("floor", "regression", "observed score is below the required minimum"),
        ("ceiling", "regression", "observed score is above the permitted maximum"),
        ("missing", "insufficient_evidence", "missing a baseline or subject result"),
        ("loss", "regression", "observed change was outside the range"),
    ],
)
def test_captured_causes_appear_in_opening_and_metric_without_repeating_heading(
    scenario, decision, cause
):
    pack, report = captured(scenario)
    before = dict(pack.files)
    original = deepcopy(report)
    view = _view(report, pack)
    assert view.decision == decision
    assert cause in view.summary and cause in view.metrics[0].explanation
    assert "The policy was not met" not in view.summary
    assert "More evidence is needed:" not in view.summary
    if scenario == "loss":
        assert "below the required minimum" in view.summary
    for rendered in (render_html(view), render_markdown(view)):
        assert decision_label(decision) in rendered
        assert cause in rendered
    assert report == original and pack.files == before


@pytest.mark.parametrize("scenario", ["pass", "count", "loss"])
def test_comparison_only_descriptions_do_not_invent_missing_policy(scenario):
    _, report = captured(scenario)
    view = _view(report, None)
    assert "policy thresholds are unavailable" in view.summary
    assert "not been independently replayed" in view.summary
    assert "interval is within the required range" not in view.summary
    assert "exceeded the policy limit" not in view.summary
    assert view.decision == report["decision"]


def test_captured_summary_uses_recorded_interval_mass_and_unknown_mass_stays_unknown():
    pack, report = captured("pass")
    report["metrics"][0]["interval"]["mass"] = 0.9
    metrics = _metric_views(report, {})
    assert "The 90% interval" in _captured_summary(metrics, report)
    assert "The 95% interval" not in _captured_summary(metrics, report)
    unknown = _captured_summary(metrics)
    assert "The uncertainty interval" in unknown
    assert "95%" not in unknown


@pytest.mark.parametrize(
    "decision", ["pass", "fail", "regression", "insufficient_evidence", "unknown"]
)
def test_browser_title_and_visible_heading_use_same_safe_decision_label(decision):
    pack, report = captured("pass")
    view = replace(
        _view(report, pack),
        decision=decision,
        title='Report <script>alert("x")</script>',
    )
    html = render_html(view)
    label = decision_label(decision)
    assert f"<title>{label} · {escape(view.title)}</title>" in html
    assert f'<h1 id="decision">{label}</h1>' in html
    assert '<script>alert("x")</script>' not in html


@pytest.mark.parametrize("prefix", ["Deterministic · ", "Judge · "])
def test_combined_metric_labels_keep_identity_while_formatting_component_name(prefix):
    from invarlock.report_presentation import display_label

    pack, report = captured("pass")
    original = _view(report, pack).metrics[0]
    metric = replace(original, name=prefix + "grounded_qa-quality")
    assert metric.name == prefix + "grounded_qa-quality"
    assert metric.display_name == prefix + "Grounded QA quality"
    assert display_label(prefix + "NLL") == prefix + "NLL"
    assert display_label(prefix + "An authored label") == prefix + "An authored label"
    assert display_label(prefix) == prefix
    nested = prefix * 2000 + "raw_label"
    assert display_label(nested) == nested


def test_raw_failure_reasons_remain_in_details_instead_of_repeating_description():
    pack, report = captured("loss")
    view = _view(report, pack)
    assert view.technical["metrics"][0]["reasons"] == report["metrics"][0]["reasons"]
    assert not any(
        note.startswith("Recorded reasons:") for note in view.metrics[0].notes
    )
