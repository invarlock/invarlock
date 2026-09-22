"""Judge descriptions explain outcomes without upgrading inconclusive evidence."""

import hashlib
from html import unescape

import pytest

from invarlock.judge_measurements.reporting import _snapshot, _view
from invarlock.report_presentation import render_html, render_markdown
from tests.judge_measurements.test_evidence_acceptance import _publish


@pytest.mark.parametrize("direction", ["higher", "lower"])
@pytest.mark.parametrize("role", ["required", "advisory"])
@pytest.mark.parametrize(
    "outcome",
    [
        "pass",
        "paired_regression",
        "subject_regression",
        "missing",
        "units",
        "precision",
        "paired_crossing",
        "subject_crossing",
    ],
)
def test_description_explains_recorded_cause_and_keeps_exact_evidence(
    tmp_path, direction, role, outcome
):
    good = int(direction == "higher")
    bad = 1 - good
    options = {"role": role, "baseline": good, "subject": good}
    policy = {
        "direction": direction,
        "allowed_degradation": "1",
        "subject_bound": "0.5",
    }
    if outcome == "paired_regression":
        options["subject"] = bad
        policy.update(allowed_degradation="0", subject_bound=None)
    elif outcome == "subject_regression":
        options.update(baseline=bad, subject=bad)
    elif outcome == "missing":
        options["incomplete"] = True
    elif outcome == "units":
        policy["minimum_units"] = 20
    elif outcome == "precision":
        policy["maximum_interval_width"] = "0.1"
    elif outcome == "paired_crossing":
        policy.update(allowed_degradation="0", subject_bound=None)
    elif outcome == "subject_crossing":
        policy["subject_bound"] = "0.731" if direction == "higher" else "0.269"
    publication, _ = _publish(tmp_path, policy_changes=policy, **options)
    before = {
        p.relative_to(publication.path): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in publication.path.rglob("*")
        if p.is_file()
    }
    retained, artifacts = _snapshot(publication.path)
    view, facts = _view(retained, artifacts)
    metric = view.metrics[0]
    description = metric.explanation
    if role == "advisory":
        assert description.startswith("This metric is advisory")
        assert view.summary.startswith("This metric is advisory")
        assert "advisory" in view.title
        assert view.family.startswith("Advisory")
    if outcome == "pass":
        assert view.decision == "pass"
        assert "interval for the score change stays within" in description
        assert (
            "at or above the required minimum"
            if direction == "higher"
            else "at or below the required maximum"
        ) in description
    elif outcome == "paired_regression":
        assert view.decision == "regression"
        assert "loss beyond the allowance of 0 score points" in description
        assert "subject's score interval" not in description
    elif outcome == "subject_regression":
        assert view.decision == "regression"
        assert (
            "entirely below the required minimum"
            if direction == "higher"
            else "entirely above the required maximum"
        ) in description
        assert "loss beyond" not in description
    else:
        assert view.decision == "insufficient_evidence"
        expected = {
            "missing": "Only 31 of 32 planned ratings completed",
            "units": "16 independent units; the policy requires at least 20",
            "precision": "wider than the permitted 0.1 score points",
            "paired_crossing": "includes losses both within and beyond the policy allowance",
            "subject_crossing": "includes values that meet the required",
        }[outcome]
        assert expected in description
        assert (
            "entirely below" not in description and "entirely above" not in description
        )
        if outcome == "missing":
            assert metric.interval is None
            assert metric.candidate == "Unavailable"
        elif outcome == "precision":
            assert metric.interval is not None
            assert "Unavailable" not in description
    assert ("advisory and does not gate required decisions" in description) == (
        role == "advisory"
    )
    assert view.summary.count("advisory and does not gate required decisions") == int(
        role == "advisory"
    )
    for output in (render_html(view), render_markdown(view)):
        for sentence in description.split(". "):
            assert sentence in unescape(output)
    for check in metric.checks:
        for reason in facts["analysis"]["reasons"]:
            assert reason not in check.explanation
    assert facts["analysis"] == publication.analysis_result.to_dict()
    assert before == {
        p.relative_to(publication.path): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in publication.path.rglob("*")
        if p.is_file()
    }
