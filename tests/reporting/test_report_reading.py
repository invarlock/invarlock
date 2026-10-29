"""Simple readings preserve scoring units, uncertainty and missing evidence."""

from dataclasses import replace
from pathlib import Path

import pytest

from invarlock.client_review import _view
from invarlock.client_review_presentation import _score_chart, brief
from invarlock.report_presentation import (
    AuthoredReview,
    CheckView,
    IntervalView,
    MetricView,
    ReportView,
    ScoreView,
)
from invarlock.report_reading import (
    observed_change,
    range_reading,
    score_caption,
    score_scale,
    suggested_action,
)

ROOT = Path(__file__).resolve().parents[2]


def metric(**changes):
    value = MetricView(
        name="Word match",
        scope="overall",
        decision="pass",
        baseline="5%",
        candidate="6%",
        change="+1 pp",
        count="400",
        explanation="Recorded comparison",
        scores=ScoreView("accuracy", 0.05, 0.06, "higher"),
        interval=IntervalView(
            -1.444, 3.526, 1, -2, "Paired interval", "pp", "minimum", 0
        ),
    )
    return replace(value, **changes)


def test_low_accuracy_uses_full_scale_without_invented_match_counts():
    value = metric()
    chart = _score_chart(value)
    assert score_scale(value) == 1
    assert "width:5.000000%" in chart and "width:6.000000%" in chart
    assert "0–100%" in chart
    assert "matched" not in chart  # A rounded mean does not supply integer counts.
    assert "includes no change" in range_reading(value)
    assert "does not establish an improvement" in range_reading(value)
    assert observed_change(value) == "+1 percentage points"


@pytest.mark.parametrize(
    "scores",
    [
        None,
        ScoreView("score", 0.2, 0.3, "higher"),
        ScoreView("accuracy", None, 0.3, "higher"),
        ScoreView("accuracy", 0.2, float("nan"), "higher"),
        ScoreView("accuracy", 0.2, float("inf"), "higher"),
        ScoreView("accuracy", -0.2, 0.3, "higher"),
        ScoreView("accuracy", 0.2, 1.1, "higher"),
        ScoreView("loss", 0, 0, "lower"),
    ],
)
def test_unavailable_or_unknown_scale_uses_text(scores):
    value = metric(scores=scores)
    assert score_scale(value) is None
    assert _score_chart(value) == ""


def test_loss_has_zero_origin_and_relative_change_without_quality_claim():
    value = metric(
        scores=ScoreView("loss", 0.5, 0.55, "lower"),
        interval=IntervalView(
            1.08, 1.12, 1.1, 1.05, "Resampling", "ratio", "maximum", 1
        ),
        decision="regression",
        checks=(CheckView("Maximum NLL ratio", "1.12", "<= 1.05", False),),
    )
    assert score_scale(value) == 0.55
    assert observed_change(value) == "+10% loss"
    reading = range_reading(value)
    assert "Loss change range: +8 to +12%" in reading
    assert "upper end at most +5%" in reading
    assert "quality" not in reading
    assert "observed change is outside" in suggested_action(value)
    within = replace(value, interval=replace(value.interval, estimate=1.02))
    assert "observed score alone is not enough" in suggested_action(within)


def test_judge_rubric_is_never_rendered_as_accuracy():
    value = metric(
        scores=ScoreView("rubric", 0.2, 0.3, "higher"),
        baseline="0.2 score",
        candidate="0.3 score",
        change="+0.1 score",
        interval=IntervalView(
            -0.1, 0.3, 0.1, -0.2, "Bounded effect", "score", "minimum", 0
        ),
    )
    assert score_scale(value) == 1
    assert "not a percentage of correct answers" in score_caption(value)
    assert "0–1" in _score_chart(value)
    assert "percentage points" not in range_reading(value)
    assert observed_change(value) == "+0.1 score"


@pytest.mark.parametrize(
    "name,phrase",
    [
        ("Complete paired results", "missing results"),
        ("Schedule completeness", "missing results"),
        ("Independent units", "independent evidence"),
        ("Interval width", "too wide or unavailable"),
        ("Subject minimum", "candidate's task quality"),
        ("Baseline accuracy", "reference version"),
        ("Policy thresholds", "unavailable or unmet checks"),
    ],
)
def test_actions_follow_the_unresolved_check(name, phrase):
    value = metric(
        decision="insufficient_evidence",
        checks=(CheckView(name, "Unavailable", "Required", None),),
    )
    assert phrase in suggested_action(value)
    assert "approve" not in suggested_action(value)


def test_missing_and_tiny_ranges_do_not_turn_into_certainty_or_zero():
    assert observed_change(metric(change="Unavailable")) == "Unavailable"
    assert "No change range is available" in range_reading(metric(interval=None))
    value = metric(
        interval=IntervalView(
            -0.000001, 0.000001, 0, -0.01, "Paired", "pp", "minimum", 0
        )
    )
    assert "-1e-06 to +1e-06" in range_reading(value)
    assert "unavailable in this view" in range_reading(
        replace(value, interval=replace(value.interval, upper=float("nan")))
    )


@pytest.mark.parametrize("family", ["native", "captured", "judge", "combined"])
def test_validated_families_supply_semantics_and_fallback_links(tmp_path, family):
    if family == "native":
        path = ROOT / "examples/acceptance-handoff/golden/evidence"
        expected = {"accuracy"}
    elif family == "captured":
        path = (
            ROOT / "examples/captured-results/references/mistral-7b-likelihood/evidence"
        )
        expected = {"loss"}
    elif family == "judge":
        from tests.judge_measurements.test_evidence_acceptance import _publish

        path = _publish(tmp_path)[0].path
        expected = {"rubric"}
    else:
        from tests.evidence_sets.test_verification import fixture

        path = fixture(tmp_path)[0]
        expected = {"accuracy", "rubric"}
    view, _, digest = _view(path)
    assert {m.scores.kind for m in view.metrics} == expected
    view = replace(
        view,
        review=AuthoredReview(
            "Authored review",
            "Owner",
            (),
            "Review the result",
            "Pending owner decision",
            (),
            "Review the checks",
        ),
    )
    for html in (True, False):
        result = brief(view, digest=digest, receipt=False, html=html)
        assert "An application-specific interpretation was not supplied" in result
        assert "Pending owner decision" in result
        assert "Not performed for this recipient" in result
        for i in range(1, len(view.metrics) + 1):
            assert f"technical-report.html#metric-result-{i}" in result


def test_bounded_brief_does_not_silently_hide_additional_results():
    view = ReportView(
        "Review",
        "Custom",
        "pass",
        "summary",
        (metric(),) * 7,
        (),
        review=AuthoredReview(
            "Authored", "Owner", (), "Review", "Pending", (), "Review"
        ),
    )
    for html in (True, False):
        result = brief(view, digest="sha256:abc", receipt=False, html=html)
        assert "Showing 6 of 7" in result
        assert "technical-report.html#metric-result-6" in result
        assert "technical-report.html#metric-result-7" not in result
