"""Opening explanations follow policy outcomes, not the direction of a score alone."""

from copy import deepcopy

import pytest

from invarlock.evidence_pack_contract import build_comparison_report
from invarlock.evidence_reporting import _report_view
from invarlock.report_presentation import render_html, render_markdown
from tests.reporting.schema.test_evidence_contract_edges import _pairs


def comparison(outcomes, *, metric="exact_match", **requirements):
    return build_comparison_report(
        comparison_id="opening-comparison",
        paired_records={
            "format": "invarlock/paired-records-v1",
            "metric": metric,
            "schedule_sha256": "0" * 64,
            **(
                {"derived_measurements": _pairs(metric=metric)["derived_measurements"]}
                if metric == "normalized_nll_per_utf8_byte"
                else {}
            ),
            "records": [
                {
                    "record_id": str(i),
                    "baseline": {"score": float(b)},
                    "subject": {"score": float(s)},
                }
                for i, (b, s) in enumerate(outcomes)
            ],
        },
        policy={"resolved_policy": {"metrics": {metric: requirements}}},
        policy_digest="sha256:" + "a" * 64,
    )


def view_of(report):
    before = deepcopy(report)
    view = _report_view(report, evidence_signer="example", observations=[])
    assert view.decision == report["verdict"]
    assert report == before
    return view


def test_observed_loss_inside_allowance_is_not_proof_of_excessive_loss():
    report = comparison(
        [(1, 1)] * 92 + [(0, 0)] * 33 + [(1, 0)] * 2 + [(0, 1)],
        delta_min_pp=-2,
        minimum_record_count=128,
        maximum_interval_width_pp=20,
    )
    assert report["comparison"]["value"] == -0.78125
    assert report["uncertainty"]["lower"] < -2 < report["comparison"]["value"]
    view = view_of(report)
    assert view.decision == "fail"
    lead = "The observed loss was within the allowance, but a loss greater than 2 percentage points could not be ruled out."
    assert view.summary.startswith(lead)
    assert "93 of 128 cases" in view.summary and "94 for the baseline" in view.summary
    for rendered in (render_html(view), render_markdown(view)):
        assert lead in rendered
        assert "Policy not met" in rendered


@pytest.mark.parametrize(
    "outcomes,minimum,verdict,opening",
    [
        (
            [(1, 1)] * 400,
            -2,
            "pass",
            "The comparison met every recorded policy requirement.",
        ),
        (
            [(1, 0)] * 128,
            -2,
            "fail",
            "The observed loss exceeded the policy allowance.",
        ),
        (
            [(1, 1)],
            -2,
            "fail",
            "The observed score did not decline, but a loss beyond the policy allowance could not be ruled out.",
        ),
        (
            [(1, 1)],
            0,
            "fail",
            "The observed score did not decline, but a loss beyond the policy allowance could not be ruled out.",
        ),
        (
            [(0, 1)],
            -2,
            "fail",
            "The observed score did not decline, but a loss beyond the policy allowance could not be ruled out.",
        ),
        (
            [(1, 1)] * 128,
            2,
            "fail",
            "The comparison did not establish the improvement required by the policy.",
        ),
        (
            [(0, 1)] * 128,
            2,
            "pass",
            "The comparison met every recorded policy requirement.",
        ),
    ],
)
def test_exact_openings_cover_success_loss_uncertainty_and_required_improvement(
    outcomes, minimum, verdict, opening
):
    view = view_of(comparison(outcomes, delta_min_pp=minimum))
    assert view.decision == verdict
    assert view.summary.startswith(opening)


@pytest.mark.parametrize(
    "outcomes,requirements,opening",
    [
        (
            [(1, 1)] * 2,
            {"minimum_record_count": 3, "maximum_interval_width_pp": 200},
            "There were too few paired records",
        ),
        (
            [(1, 1)] * 2,
            {"minimum_record_count": 2, "maximum_interval_width_pp": 1},
            "The uncertainty interval was wider",
        ),
        (
            [(0, 1)] * 2,
            {"minimum_side_accuracy": 0.5},
            "The baseline accuracy was below",
        ),
        (
            [(1, 0)] * 2,
            {"minimum_side_accuracy": 0.5},
            "The subject accuracy was below",
        ),
    ],
)
def test_other_failed_requirements_are_not_described_as_comparison_regression(
    outcomes, requirements, opening
):
    view = view_of(comparison(outcomes, delta_min_pp=-100, **requirements))
    assert view.decision == "fail"
    assert view.metrics[0].checks[0].passed
    assert view.summary.startswith(opening)


@pytest.mark.parametrize(
    "outcomes,maximum,verdict,opening",
    [
        (
            [(2, 2)] * 2,
            1.1,
            "pass",
            "The comparison met every recorded policy requirement.",
        ),
        (
            [(2, 2.4)] * 2,
            1.1,
            "fail",
            "The observed subject-to-baseline NLL ratio exceeded the policy limit.",
        ),
        (
            [(2, 1), (2, 3)],
            1.1,
            "fail",
            "The observed subject-to-baseline NLL ratio was within the limit, but its uncertainty interval extended beyond it.",
        ),
        (
            [(2, 1.8)] * 2,
            0.8,
            "fail",
            "The observed subject-to-baseline NLL ratio exceeded the policy limit.",
        ),
        (
            [(2, 1), (2, 2)],
            0.8,
            "fail",
            "The observed subject-to-baseline NLL ratio was within the limit, but its uncertainty interval extended beyond it.",
        ),
        (
            [(2, 1)] * 2,
            0.8,
            "pass",
            "The comparison met every recorded policy requirement.",
        ),
    ],
)
def test_nll_openings_respect_lower_is_better_and_required_improvement(
    outcomes, maximum, verdict, opening
):
    view = view_of(
        comparison(outcomes, metric="normalized_nll_per_utf8_byte", ratio_max=maximum)
    )
    assert view.decision == verdict
    assert view.summary.startswith(opening)
    assert "likelihood ratio" not in view.summary


@pytest.mark.parametrize(
    "kwargs,decision,opening",
    [
        ({}, "pass", "The judge comparison met every required policy requirement."),
        (
            {"baseline": 1, "subject": 0},
            "regression",
            "The judge comparison did not meet at least one required policy requirement.",
        ),
        (
            {"incomplete": True},
            "insufficient_evidence",
            "The available evidence does not establish that",
        ),
        (
            {"policy_changes": {"minimum_units": 20}},
            "insufficient_evidence",
            "The available evidence does not establish that",
        ),
        (
            {"role": "advisory"},
            "pass",
            "The judge comparison met every advisory policy requirement.",
        ),
    ],
)
def test_judge_openings_distinguish_decisions_and_advisory_scope(
    tmp_path, kwargs, decision, opening
):
    from invarlock.judge_measurements.reporting import _snapshot, _view
    from tests.judge_measurements import test_evidence_acceptance as support

    publication, _ = support._publish(tmp_path, **kwargs)
    before = {p.name: p.read_bytes() for p in publication.path.iterdir() if p.is_file()}
    retained, artifacts = _snapshot(publication.path)
    view, _ = _view(retained, artifacts)
    assert view.decision == decision
    assert view.summary.startswith(opening)
    if kwargs.get("role") == "advisory":
        assert "does not gate required decisions" in view.summary
    if kwargs.get("incomplete"):
        assert "unavailable for the incomplete schedule" in view.summary
    assert {
        p.name: p.read_bytes() for p in publication.path.iterdir() if p.is_file()
    } == before


def test_captured_success_is_explicit_before_the_observed_values():
    from invarlock.evaluation_records.templates import example_project
    from invarlock.record_reporting import _view
    from tests._evaluation_support import build_pack, pack_json

    baseline, subject, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    policy["metrics"][0].update(
        maximum_regression=1, maximum_interval_width=2, subject_minimum=0
    )
    snapshot = build_pack(baseline, subject, policy)
    report = pack_json(snapshot, "report")
    before = deepcopy(report)
    view = _view(report, snapshot)
    assert view.metrics[0].decision == "pass"
    assert view.summary.startswith("This result met every recorded policy requirement.")
    assert "Across" in view.summary
    assert report == before
