"""Combined descriptions retain each required component's distinct outcome."""

from dataclasses import replace
from itertools import product

import pytest

from invarlock.evidence_sets import reporting
from invarlock.report_presentation import decision_label, render_html, render_markdown
from tests.evidence_sets.test_verification import fixture


@pytest.mark.parametrize(
    ("deterministic", "judge"),
    tuple(product(("pass", "regression", "insufficient_evidence"), repeat=2)),
)
def test_opening_names_both_component_outcomes_without_changing_them(
    tmp_path, monkeypatch, deterministic, judge
):
    root, _ = fixture(tmp_path)
    captured_view = reporting.captured_view
    judge_view = reporting.judge_view

    def captured(*args, **kwargs):
        view = captured_view(*args, **kwargs)
        return replace(
            view,
            decision=deterministic,
            metrics=tuple(
                replace(
                    m,
                    decision=deterministic,
                    explanation="Retained deterministic explanation.",
                )
                for m in view.metrics
            ),
        )

    def judged(*args, **kwargs):
        view, facts = judge_view(*args, **kwargs)
        return replace(
            view,
            decision=judge,
            metrics=tuple(
                replace(m, decision=judge, explanation="Retained judge explanation.")
                for m in view.metrics
            ),
        ), facts

    monkeypatch.setattr(reporting, "captured_view", captured)
    monkeypatch.setattr(reporting, "judge_view", judged)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    view, facts, _ = reporting.build_evidence_set_view(root)
    expected = (
        "regression"
        if "regression" in (deterministic, judge)
        else "insufficient_evidence"
        if "insufficient_evidence" in (deterministic, judge)
        else "pass"
    )
    assert view.title == "InvarLock combined comparison report"
    assert view.decision == facts["decision"] == expected
    assert facts["component_decisions"] == {
        "deterministic": deterministic,
        "judge": judge,
    }
    assert [metric.decision for metric in view.metrics] == [deterministic, judge]
    for rendered in (render_html(view), render_markdown(view)):
        assert (
            f"Deterministic comparison: {decision_label(deterministic).lower()}."
            in rendered
        )
        assert f"Judge comparison: {decision_label(judge).lower()}." in rendered
        assert "Both components must satisfy their recorded policies" in rendered
        assert "same frozen baseline and subject answers" in rendered
        assert "no joint confidence guarantee" in rendered
    assert facts["recipient_acceptance"] == "not_performed"
    assert before == {
        path: path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_shared_context_shown_once_without_hiding_different_component_values(
    tmp_path, monkeypatch
):
    root, _ = fixture(tmp_path)
    original = reporting.judge_view

    def additional_context(*args, **kwargs):
        view, facts = original(*args, **kwargs)
        return replace(
            view,
            context=(*view.context, ("Baseline evaluator", "Another recorded value")),
        ), facts

    monkeypatch.setattr(reporting, "judge_view", additional_context)
    view, _, _ = reporting.build_evidence_set_view(root)
    assert len(view.context) == len(set(view.context))
    assert view.context.count(("Baseline evaluator", "fixture 1")) == 1
    assert ("Baseline evaluator", "Another recorded value") in view.context
    assert any(name == "Judge provider" for name, _ in view.context)
    assert "recipient-owned key" in dict(view.assurance)["Deterministic signature"]
    assert "Signature present" in dict(view.assurance)["Judge signature"]


def test_real_incomplete_judging_explains_the_cause_before_component_charts(tmp_path):
    root, _ = fixture(tmp_path, incomplete=True)
    view, facts, _ = reporting.build_evidence_set_view(root)
    assert view.decision == "insufficient_evidence"
    assert facts["component_decisions"]["deterministic"] == "pass"
    assert "Judge finding for" in view.summary
    assert "Only" in view.summary
    assert "planned ratings completed" in view.summary
    assert "paired comparison could not be calculated" in view.summary
