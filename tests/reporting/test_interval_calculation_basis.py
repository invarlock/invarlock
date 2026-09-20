"""Interval explanations follow retained methods, counts and error budgets."""

from copy import deepcopy
from functools import partial

import pytest

from invarlock.evidence_pack_contract import build_comparison_report
from invarlock.evidence_reporting import _report_view
from invarlock.judge_measurements.reporting import _snapshot, _view
from invarlock.record_reporting import _metric_views
from invarlock.report_presentation import render_html, render_markdown
from tests.evidence_packs.test_evidence_reporting import _report
from tests.judge_measurements import test_evidence_acceptance as support
from tests.reporting.schema.test_evidence_contract_edges import _pairs
from tests.reporting.test_captured_summary import binary_comparison


def test_native_exact_basis_describes_paired_counts_and_percentage_points():
    report = _report()
    before = deepcopy(report)
    view = _report_view(report, evidence_signer="example", observations=[])
    interval = view.metrics[0].interval
    assert interval.method == "Newcombe hybrid score"
    assert "same 2 cases" in interval.basis[0]
    assert interval.basis[1] == (
        "Both matched: 1; baseline only: 0; subject only: 1; neither matched: 0."
    )
    for rendered in (render_html(view), render_markdown(view)):
        assert "How this interval was calculated" in rendered
        assert "nominal 95% confidence interval" in rendered
        assert "not relative percent change" in rendered
        assert "not the share of correct answers" in rendered
    assert report == before


def test_native_nll_basis_is_fixed_schedule_resampling_of_a_ratio():
    report = build_comparison_report(
        comparison_id="nll",
        paired_records=_pairs(
            metric="normalized_nll_per_utf8_byte", baseline=2.0, subject=2.2
        ),
        policy={
            "resolved_policy": {
                "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 1.2}}
            }
        },
        policy_digest="sha256:" + "a" * 64,
    )
    before = deepcopy(report)
    interval = (
        _report_view(report, evidence_signer="example", observations=[])
        .metrics[0]
        .interval
    )
    assert "same 2 paired records" in interval.basis[0]
    assert "resampling count is 2,048" in interval.basis[0]
    assert "subject-to-baseline mean NLL ratio" in interval.basis[1]
    assert "not a population confidence interval" in interval.basis[2]
    assert report == before


def test_captured_binary_basis_uses_paired_method_without_inventing_counts():
    _, comparison, _ = binary_comparison()
    before = deepcopy(comparison)
    interval = _metric_views(comparison, {})[0].interval
    assert "Newcombe hybrid score" in interval.basis[0]
    assert "Both matched" not in " ".join(interval.basis)
    assert "not relative percent change" in interval.basis[1]
    assert "simultaneous confidence guarantee" in interval.basis[2]
    assert comparison == before


def test_captured_resampling_basis_uses_retained_count_and_mass():
    from invarlock.evaluation_comparison.comparison import compare_runs
    from tests.evaluation_comparison.test_likelihood import policy, row, run

    configured = policy()
    comparison = compare_runs(run([row()]), run([row(logprob=-3)]), configured)
    # A comparison-only projection reports its supplied metadata; it does not replay it.
    comparison["metrics"][0]["interval"].update(replicates=512, mass=0.9)
    before = deepcopy(comparison)
    interval = _metric_views(comparison, {})[0].interval
    assert "resampling count is 512" in interval.basis[0]
    assert "central 90%" in interval.basis[1]
    assert "mean NLL ratio" in interval.basis[1]
    assert interval.label.startswith("90%")
    assert "not a population confidence interval" in interval.basis[2]
    assert comparison == before


def test_captured_unknown_method_does_not_invent_a_calculation():
    _, comparison, _ = binary_comparison()
    comparison["metrics"][0]["interval"]["method"] = "other-declared-method"
    interval = _metric_views(comparison, {})[0].interval
    assert interval.method == ""
    assert interval.basis[0] == "Retained interval method: other-declared-method."
    assert "explanation is unavailable" in interval.basis[1]
    assert "resampl" not in " ".join(interval.basis)


@pytest.mark.parametrize(
    ("alpha", "family", "confidence"), [("0.1", 4, "90"), ("0.2", 5, "80")]
)
def test_judge_basis_uses_independent_units_and_actual_family_budget(
    tmp_path, monkeypatch, alpha, family, confidence
):
    monkeypatch.setattr(
        support,
        "_bundle",
        partial(support._bundle, groups=("a", "a", "b", "b"), repetitions=3),
    )
    publication, _ = support._publish(
        tmp_path, policy_changes={"alpha": alpha, "comparison_family_size": family}
    )
    before = {
        p.relative_to(publication.path): p.read_bytes()
        for p in publication.path.rglob("*")
        if p.is_file()
    }
    retained, artifacts = _snapshot(publication.path)
    view, facts = _view(retained, artifacts)
    interval = view.metrics[0].interval
    assert "4 cases grouped into 2 independent units" in interval.basis[0]
    assert "3 ratings per side" in interval.basis[0]
    assert "normalized rubric score points" in interval.basis[1]
    assert f"({alpha} / {family})" in interval.basis[2]
    assert f"at least {confidence}%" in interval.basis[2]
    assert "95%" not in " ".join(interval.basis)
    assert "fixed benchmark" in interval.basis[3]
    for rendered in (render_html(view), render_markdown(view)):
        assert "How this interval was calculated" in rendered
        assert f"at least {confidence}%" in rendered
    assert facts["analysis"] == publication.analysis_result.to_dict()
    assert before == {
        p.relative_to(publication.path): p.read_bytes()
        for p in publication.path.rglob("*")
        if p.is_file()
    }


def test_captured_fractional_score_basis_keeps_score_units():
    from invarlock.evaluation_records.templates import example_project
    from invarlock.record_reporting import _view as captured_view
    from tests._evaluation_support import build_pack, pack_json

    baseline, subject, policy = example_project("extraction")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    for row in subject["records"]:
        row["output"]["currency"] = "CAD"
    snapshot = build_pack(baseline, subject, policy)
    comparison = pack_json(snapshot, "report")
    view = captured_view(comparison, snapshot)
    interval = view.metrics[0].interval
    assert "resampling count is 2,048" in interval.basis[0]
    assert "mean score change in score units" in interval.basis[1]
    assert "percentage points" not in " ".join(interval.basis)
    assert "not a population confidence interval" in interval.basis[2]
