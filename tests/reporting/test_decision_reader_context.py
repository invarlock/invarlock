"""Reports expose decision-relevant values without changing evidence semantics."""

import re
from functools import partial
from html import unescape

import pytest

from invarlock.evidence_reporting import _report_view
from invarlock.judge_measurements.reporting import _snapshot, _view
from invarlock.report_presentation import render_html, render_markdown
from tests.evidence_packs.test_evidence_reporting import _report
from tests.judge_measurements import test_evidence_acceptance as support


def test_native_summary_explains_observed_result_and_exact_match_counts():
    report = _report()
    view = _report_view(report, evidence_signer="demo", observations=[])
    metric = view.metrics[0]
    assert "100%" in view.summary and "50%" in view.summary
    assert "+50 pp" in view.summary
    assert metric.baseline_detail == "1 of 2 matched"
    assert metric.candidate_detail == "2 of 2 matched"
    assert view.technical is report
    for output in (render_html(view), render_markdown(view)):
        rendered = unescape(re.sub(r"<[^>]+>", "", output))
        assert "1 of 2 matched" in rendered and "2 of 2 matched" in rendered
        assert "independent acceptance requires" in rendered


@pytest.mark.parametrize("incomplete", [False, True])
def test_judge_does_not_count_repetitions_or_clustered_cases_as_independent_units(
    tmp_path, monkeypatch, incomplete
):
    monkeypatch.setattr(
        support,
        "_bundle",
        partial(support._bundle, groups=("a", "a", "b", "b"), repetitions=3),
    )
    publication, _ = support._publish(tmp_path, incomplete=incomplete)
    retained, artifacts = _snapshot(publication.path)
    view, facts = _view(retained, artifacts)
    metric = view.metrics[0]
    assert metric.count_label == "Complete independent units"
    assert metric.count == ("1" if incomplete else "2")
    assert metric.count_detail == (
        "2 scheduled units; 4 cases; 23/24 completed trials"
        if incomplete
        else "2 scheduled units; 4 cases; 24/24 completed trials"
    )
    assert facts["analysis"] == publication.analysis_result.to_dict()
    assert view.decision == publication.analysis_result.decision
    for output in (render_html(view), render_markdown(view)):
        rendered = unescape(re.sub(r"<[^>]+>", "", output))
        assert "not percentages of correct answers" in rendered
        assert metric.count_detail in rendered
        if incomplete:
            assert "unavailable for the incomplete schedule" in rendered
            assert metric.baseline == metric.candidate == metric.change == "Unavailable"
        else:
            assert "0 score" in rendered and "1 score" in rendered
            assert "a change of +1 score" in rendered
            assert "Across 2 complete independent units" in rendered
