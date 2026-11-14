"""Recipient-facing semantic regressions from retained report failures."""

import re
from html import unescape

import pytest

from invarlock.client_review import build_client_review
from invarlock.client_review_presentation import brief
from invarlock.report_presentation import (
    AuthoredReview,
    CheckView,
    IntervalView,
    ReportView,
)
from invarlock.report_reading import range_reading, suggested_action
from tests.reporting.test_client_handoff import REFERENCE, context
from tests.reporting.test_report_reading import metric


def visible(raw):
    return unescape(re.sub("<[^>]*>", " ", raw))


def test_restrictions_and_packaged_sources_survive(tmp_path):
    restriction = "NOT APPROVED FOR CUSTOMER TRAFFIC."
    ctx = context(
        tmp_path,
        decision="Limited exercise. " + "Rationale. " * 340 + restriction,
        context=[
            ["First", "One"],
            ["Second", "Two"],
            ["Excluded use", restriction],
            ["Comparison", "NO CUSTOMER USE FROM AUTHORED COMPARISON."],
        ],
    )
    out = tmp_path / "package"
    build_client_review(REFERENCE / "evidence", context_path=ctx, output=out)
    for name in ("brief.html", "brief.md", "review.html", "review.md"):
        assert restriction in visible((out / name).read_text())
        assert "NO CUSTOMER USE FROM AUTHORED COMPARISON." in visible(
            (out / name).read_text()
        )
    assert "Excluded use" in (out / "brief.html").read_text()
    assert (
        "evidence/reports/evaluation.report.json#/metrics/0"
        in (out / "technical-report.html").read_text()
    )
    assert "footer{display:none}" not in (out / "brief.html").read_text()


@pytest.mark.parametrize("reverse", [False, True])
def test_all_blockers_precede_passes_and_keep_original_ids(reverse):
    passes = [metric(name=f"Passing {i}") for i in range(7)]
    blockers = [
        metric(
            name=f"Blocker {i}",
            decision="fail",
            checks=(CheckView("Subject minimum", "0", ">= 1", False),),
        )
        for i in range(8)
    ]
    metrics = tuple(blockers + passes if reverse else passes + blockers)
    view = ReportView(
        "Review",
        "combined",
        "fail",
        "summary",
        metrics,
        (),
        review=AuthoredReview("Legacy", "Owner", (), "Review", "Pending", (), "Review"),
    )
    for html in (True, False):
        text = brief(view, digest="sha256:abc", receipt=False, html=html)
        for i, m in enumerate(metrics, 1):
            if m.decision != "pass":
                assert m.name in text
                assert f"technical-report.html#metric-result-{i}" in text
        assert (
            text.index("Blocker 0") < text.find("Passing 0")
            if "Passing 0" in text
            else True
        )


@pytest.mark.parametrize(
    "direction,bound,threshold",
    [
        ("maximum", 1.0980647428628072, 1.0980646428628071),
        ("minimum", -0.0000002, -0.0000001),
    ],
)
def test_near_threshold_failure_has_exact_margin(direction, bound, threshold):
    interval = IntervalView(
        bound, bound, bound, threshold, "Range", "score", direction, 0
    )
    text = range_reading(metric(decision="fail", interval=interval))
    assert "Failure margin" in text
    assert repr(abs(bound - threshold)) in text


def test_multiple_unmet_requirements_all_supply_followup():
    m = metric(
        decision="fail",
        checks=(
            CheckView("Interval width", "1", "<= .1", False),
            CheckView("Subject minimum", "0", ">= .5", False),
        ),
    )
    text = suggested_action(m)
    assert "uncertainty" in text and "candidate's task quality" in text
    assert "Interval width" in text and "Subject minimum" in text


def test_structured_exception_and_generated_facts(tmp_path):
    ctx = context(
        tmp_path,
        format="invarlock/client-review-context-v2",
        intended_use="Internal evaluation",
        excluded_uses="No customer traffic",
        decision_conditions="Approval expires after internal exercise",
        decision_state="accepted",
        exception_rationale="Owner accepts a limited internal trial despite the failed loss bound",
        summary=[["Untrusted interpretation", "Every requirement passed."]],
    )
    out = tmp_path / "structured"
    build_client_review(REFERENCE / "evidence", context_path=ctx, output=out)
    for name in [
        "brief.html",
        "brief.md",
        "review.html",
        "review.md",
        "technical-report.html",
        "technical-report.md",
    ]:
        text = visible((out / name).read_text())
        assert "Generated facts" in text
        assert "Policy not met" in text
        assert "No customer traffic" in text
        assert "exception" in text
        assert "No absolute minimum score" in text
        assert "Observation window" in text and "Verification time" in text


@pytest.mark.parametrize(
    "change",
    [
        {"decision_state": "made-up"},
        {"recipient_verification": "accepted"},
        {"exception_rationale": ""},
    ],
)
def test_structured_invalid_or_unacknowledged_acceptance_fails(tmp_path, change):
    from invarlock.client_review import ClientReviewError

    values = {
        "format": "invarlock/client-review-context-v2",
        "intended_use": "Internal",
        "excluded_uses": "External",
        "decision_conditions": "Review first",
        "decision_state": "accepted",
    }
    values.update(change)
    with pytest.raises(ClientReviewError):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path, **values),
            output=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_critical_facts_precede_long_authored_narrative(tmp_path):
    ctx = context(
        tmp_path,
        decision="LONG OWNER NARRATIVE. " * 120,
        summary=[["Interpretation", "LONG INTERPRETATION. " * 20]],
    )
    out = tmp_path / "priority"
    build_client_review(REFERENCE / "evidence", context_path=ctx, output=out)
    for name in ["brief.html", "brief.md"]:
        text = (out / name).read_text()
        assert text.index("Recipient verification") < text.index("LONG OWNER NARRATIVE")
        assert text.index("Maximum NLL ratio") < text.index("LONG OWNER NARRATIVE")
        assert text.index("Results at a glance") < text.index("LONG INTERPRETATION")


def test_case_examples_include_population_and_all_categories():
    from invarlock.client_review import _view
    from tests.reporting.test_client_handoff import ROOT

    view, _, _ = _view(
        ROOT / "examples/hosted-service/references/mistral-7b-http/evidence"
    )
    text = " ".join(v for _, v in view.explanations)
    assert "400 paired records" in text
    assert all(word in text for word in ("output changed", "unchanged", "missing"))
    assert "first record in stable ID order" in text
    assert "Attempted and excluded totals" in text


def test_recorded_case_categories_use_retained_scores_and_canonical_missingness():
    from invarlock.report_cases import captured_case_readings

    left = [
        {"id": str(i), "scores": {"quality": v}, "output": "same", "error": None}
        for i, v in enumerate([0, 1, 1, 0])
    ]
    right = [
        {"id": str(i), "scores": {"quality": v}, "output": "same", "error": None}
        for i, v in enumerate([1, 0, 1, 0])
    ]
    inputs = {
        "baseline": {"records": left},
        "subject": {"records": right},
        "policy": {
            "metrics": [
                {
                    "name": "quality",
                    "kind": "recorded",
                    "score_key": "quality",
                    "direction": "higher",
                    "unit": "score",
                }
            ]
        },
    }
    comparison = {
        "metrics": [{"name": "quality", "slice": "overall", "missing_ids": ["3"]}]
    }
    text = " ".join(v for _, v in captured_case_readings(inputs, comparison))
    assert "1 improved, 1 regressed, 1 unchanged, 1 missing" in text
    assert "Record ID 3: a retained result is missing" in text
