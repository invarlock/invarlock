from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, Inexact, localcontext
from pathlib import Path
from typing import cast

import pytest

from invarlock.evaluation_records.cases import case_set_digest
from invarlock.evaluation_records.io import run_digest
from invarlock.judge_measurement_types import JudgeMeasurementPlan, JudgeMeasurements
from invarlock.judge_measurements.analysis import (
    JudgeAnalysisPolicy,
    analyze_measurements,
    combine_analysis_results,
)
from invarlock.judge_measurements.contracts import (
    JudgeMeasurementContractError,
    canonical_payload,
    expected_trial_id,
    measurement_plan_digest,
)

D = Decimal
FIXTURES = Path(__file__).parents[1] / "fixtures" / "judge_measurements"


def _runs(plan):
    runs = []
    for side in ("baseline", "subject"):
        run = json.loads((FIXTURES / f"{side}_run.json").read_text())
        template = run["records"][0]
        run["records"] = [
            {**copy.deepcopy(template), "id": case["case_id"]}
            for case in plan["sampling"]["case_units"]
        ]
        runs.append(run)
    return runs


def _analyze(plan, data, policy):
    baseline, subject = _runs(plan)
    return analyze_measurements(
        plan, data, policy, baseline_run=baseline, subject_run=subject
    )


def _retain(data: JudgeMeasurements) -> None:
    payload = canonical_payload(
        {"format": "invarlock/retained-judge-json-v1", "trials": data["trials"]}
    )
    data["sources"][0].update(
        content=payload.decode(),
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
    )
    count = sum(trial["status"] == "complete" for trial in data["trials"])
    data["completeness"].update(
        completed_trials=count,
        status="complete"
        if count == data["completeness"]["expected_trials"]
        else "incomplete",
    )


def _bundle(
    *,
    groups: tuple[str, ...] = tuple(f"u{i}" for i in range(16)),
    repetitions: int = 1,
    baseline: int = 0,
    subject: int = 1,
) -> tuple[JudgeMeasurementPlan, JudgeMeasurements]:
    plan = cast(JudgeMeasurementPlan, json.loads((FIXTURES / "plan.json").read_text()))
    data = cast(
        JudgeMeasurements, json.loads((FIXTURES / "measurements.json").read_text())
    )
    plan["sampling"]["case_units"] = [
        {"case_id": f"case-{i}", "unit_id": group} for i, group in enumerate(groups)
    ]
    binding = plan["answer_bindings"][0]
    plan["answer_bindings"] = [
        {**binding, "case_id": case["case_id"]}
        for case in plan["sampling"]["case_units"]
    ]
    expected = len(groups) * repetitions * 2
    plan["schedule"].update(repetitions=repetitions, expected_trials=expected)
    baseline_run, subject_run = _runs(plan)
    plan["baseline_run_sha256"] = run_digest(baseline_run)
    plan["subject_run_sha256"] = run_digest(subject_run)
    plan["case_set_sha256"] = case_set_digest(
        {
            "format": "invarlock/evaluation-case-set-v1",
            "cases": [
                {key: row[key] for key in ("id", "input", "expected", "metadata")}
                for row in baseline_run["records"]
            ],
        }
    )
    digest = measurement_plan_digest(plan)
    templates = data["trials"]
    data["trials"] = []
    for case in plan["sampling"]["case_units"]:
        for template, value in zip(templates, (baseline, subject), strict=True):
            for rep in range(1, repetitions + 1):
                trial = copy.deepcopy(template)
                trial.update(
                    case_id=case["case_id"],
                    repetition=rep,
                    plan_sha256=digest,
                    trial_id=expected_trial_id(
                        digest, case["case_id"], trial["side"], rep
                    ),
                )
                trial["attempts"][0]["source"]["record_index"] = len(data["trials"])
                trial["attempts"][0]["source"]["model_event_id"] = (
                    f"event-{len(data['trials'])}"
                )
                trial["attempts"][0]["request_id"] = f"request-{len(data['trials'])}"
                rating = "correct" if value else "incorrect"
                response = canonical_payload({"rating": rating})
                trial["attempts"][0]["response"] = {
                    "media_type": "application/json",
                    "text": response.decode(),
                    "sha256": hashlib.sha256(response).hexdigest(),
                }
                trial["parse"].update(rating=rating, value=str(value))
                data["trials"].append(trial)
    data["plan_sha256"] = digest
    data["completeness"].update(expected_trials=expected, recorded_trials=expected)
    _retain(data)
    return plan, data


def _policy(**changes):
    return JudgeAnalysisPolicy(direction="higher", allowed_degradation=D(0), **changes)


def test_complete_curated_benchmark_analysis_is_bound_and_serializable():
    plan, data = _bundle()
    result = _analyze(plan, data, _policy())
    assert result.decision == "pass"
    assert result.effect_interval.mean == D(1)
    assert result.subject_interval.mean == D(1)
    assert result.subject_interval_role == "descriptive"
    assert (
        result.effect_interval.comparisons == result.subject_interval.comparisons == 2
    )
    assert result.counts.scheduled_units == result.counts.complete_units == 16
    assert result.counts.completed_trials == result.counts.expected_trials == 32
    assert result.counts.incomplete_trials == 0
    assert result.plan_sha256 == measurement_plan_digest(plan)
    assert (
        result.measurements_sha256
        == hashlib.sha256(canonical_payload(data)).hexdigest()
    )
    assert result.sampling_basis == "curated_benchmark"
    assert "No population generalization" in result.estimand
    assert "distributions may differ" in result.assumptions[0]
    encoded = result.to_dict()
    assert json.loads(json.dumps(encoded)) == encoded
    assert encoded["effect_interval"]["mean"] == "1.000000000000000"
    encoded["counts"]["complete_units"] = 0
    assert result.counts.complete_units == 16
    with pytest.raises(FrozenInstanceError):
        result.decision = "regression"


def test_repetitions_never_inflate_inference_unit_count():
    once = _analyze(*_bundle(), _policy())
    repeated = _analyze(*_bundle(repetitions=5), _policy())
    assert repeated.effect_interval == once.effect_interval
    assert repeated.subject_interval == once.subject_interval
    assert repeated.counts.completed_trials == once.counts.completed_trials * 5
    assert repeated.counts.complete_units == once.counts.complete_units


def test_declared_clusters_reduce_inference_count_and_get_equal_weight():
    clustered = _analyze(*_bundle(groups=("a",) * 15 + ("b",)), _policy())
    separate = _analyze(*_bundle(), _policy())
    assert clustered.effect_interval.unit_count == 2
    assert clustered.effect_interval.lower < separate.effect_interval.lower
    plan, data = _bundle(groups=("a", "a", "b"))
    # Subject scores in the two units become (1+0)/2 and 1, respectively.
    trial = data["trials"][3]
    response = canonical_payload({"rating": "incorrect"})
    trial["attempts"][0]["response"].update(
        text=response.decode(), sha256=hashlib.sha256(response).hexdigest()
    )
    trial["parse"].update(rating="incorrect", value="0")
    _retain(data)
    result = _analyze(plan, data, _policy())
    assert result.subject_interval.mean == D(".75")


@pytest.mark.parametrize("failure", ["invalid", "refusal", "unavailable"])
def test_incomplete_valid_schedule_has_no_inferential_intervals(failure):
    plan, data = _bundle(repetitions=2)
    trial = data["trials"][0]
    trial["status"] = "incomplete"
    trial["parse"].update(status=failure, rating=None, value=None)
    if failure == "invalid":
        response = b'{"rating":"not-in-scale"}'
        trial["attempts"][0]["response"].update(
            text=response.decode(), sha256=hashlib.sha256(response).hexdigest()
        )
    else:
        trial["selected_attempt"] = None
        trial["attempts"][0].update(
            status="refusal" if failure == "refusal" else "transport_error",
            response=None,
            error={"code": failure, "message": "retained failure"},
        )
    _retain(data)
    result = _analyze(plan, data, _policy())
    assert result.decision == "insufficient_evidence"
    assert result.effect_interval is None and result.subject_interval is None
    assert result.reasons == ("incomplete_planned_schedule",)
    assert result.counts.completed_trials == 63
    assert result.counts.incomplete_trials == 1
    assert result.counts.complete_units == 15
    assert result.gates == ()


def test_omitted_slots_and_unreplayed_parse_fail_contract_validation():
    plan, data = _bundle()
    data["trials"].pop()
    data["completeness"]["recorded_trials"] -= 1
    _retain(data)
    with pytest.raises(JudgeMeasurementContractError, match="omit"):
        _analyze(plan, data, _policy())
    plan, data = _bundle()
    data["trials"][0]["parse"]["value"] = "1"
    _retain(data)
    with pytest.raises(JudgeMeasurementContractError, match="replay"):
        _analyze(plan, data, _policy())


def test_analysis_requires_the_approved_frozen_runs():
    plan, data = _bundle()
    baseline, subject = _runs(plan)
    subject["records"][0]["output"] = "substituted answer"
    with pytest.raises(JudgeMeasurementContractError, match="subject run"):
        analyze_measurements(
            plan, data, _policy(), baseline_run=baseline, subject_run=subject
        )


def test_lower_is_better_reverses_effect_and_absolute_bound_rules():
    plan, data = _bundle(baseline=1, subject=0)
    lower = JudgeAnalysisPolicy("lower", D(0), subject_bound=D(".8"))
    result = _analyze(plan, data, lower)
    assert result.decision == "pass"
    assert result.effect_interval.mean == -1
    assert result.subject_interval_role == "decision"
    assert [gate.decision for gate in result.gates] == ["pass", "pass"]
    assert _analyze(plan, data, _policy()).decision == "regression"


def test_absolute_subject_gate_can_fail_despite_acceptable_effect():
    plan, data = _bundle(baseline=0, subject=0)
    result = _analyze(
        plan, data, JudgeAnalysisPolicy("higher", D(1), subject_bound=D(".8"))
    )
    assert [gate.decision for gate in result.gates] == ["pass", "regression"]
    assert result.decision == "regression"


def test_equality_at_outward_effect_and_subject_thresholds_passes():
    plan, data = _bundle(baseline=0, subject=0)
    first = _analyze(plan, data, _policy())
    policy = JudgeAnalysisPolicy(
        "higher",
        first.effect_interval.lower.copy_negate(),
        subject_bound=first.subject_interval.lower,
    )
    assert _analyze(plan, data, policy).decision == "pass"
    regression_boundary = replace(policy, subject_bound=first.subject_interval.upper)
    assert _analyze(plan, data, regression_boundary).decision == "insufficient_evidence"


def test_minimum_units_and_width_limit_are_insufficient_evidence():
    plan, data = _bundle()
    minimum = _analyze(plan, data, _policy(minimum_units=17))
    assert minimum.decision == "insufficient_evidence"
    assert minimum.reasons == ("minimum_units_not_met",)
    width = _analyze(plan, data, _policy(maximum_interval_width=D(".01")))
    assert width.decision == "insufficient_evidence"
    assert width.reasons == ("maximum_interval_width_exceeded",)
    initial = _analyze(plan, data, _policy())
    exact_width = initial.effect_interval.upper - initial.effect_interval.lower
    assert (
        _analyze(plan, data, _policy(maximum_interval_width=exact_width)).decision
        == "pass"
    )


def test_advisory_regression_keeps_own_result_without_overriding_required_pass():
    required = _analyze(*_bundle(), _policy(comparison_family_size=4))
    advisory = _analyze(
        *_bundle(baseline=1, subject=0),
        _policy(required=False, comparison_family_size=4),
    )
    assert advisory.decision == "regression"
    assert combine_analysis_results([required, advisory]) == "pass"
    assert combine_analysis_results([advisory]) == "insufficient_evidence"
    assert combine_analysis_results([]) == "insufficient_evidence"
    assert (
        combine_analysis_results(
            [required, replace(advisory, policy=_policy(comparison_family_size=4))]
        )
        == "regression"
    )


def test_larger_declared_family_widens_both_published_intervals():
    plan, data = _bundle()
    first = _analyze(plan, data, _policy())
    larger = _analyze(plan, data, _policy(comparison_family_size=20))
    assert larger.effect_interval.lower < first.effect_interval.lower
    assert larger.subject_interval.lower < first.subject_interval.lower


def test_combination_refuses_underdeclared_or_inconsistent_family():
    result = _analyze(*_bundle(), _policy())
    with pytest.raises(ValueError, match="too small"):
        combine_analysis_results([result, result])
    different_alpha = _analyze(
        *_bundle(), _policy(alpha=D(".01"), comparison_family_size=4)
    )
    with pytest.raises(ValueError, match="same family alpha"):
        combine_analysis_results([result, different_alpha])


@pytest.mark.parametrize(
    "changes",
    [
        {"comparison_family_size": 1},
        {"comparison_family_size": True},
        {"minimum_units": 0},
        {"maximum_interval_width": D(-1)},
        {"maximum_interval_width": D("NaN")},
        {"alpha": D(0)},
        {"alpha": D(1)},
        {"alpha": 0.05},
        {"required": 1},
        {"subject_bound": D("Infinity")},
    ],
)
def test_invalid_analysis_policies_fail(changes):
    with pytest.raises(ValueError):
        _policy(**changes)


def test_numeric_results_are_stable_under_ambient_decimal_context():
    plan, data = _bundle()
    policy = _policy()
    expected = _analyze(plan, data, policy)
    with localcontext() as ctx:
        ctx.prec = 2
        ctx.traps[Inexact] = True
        assert _analyze(plan, data, policy) == expected


def test_decimal_string_scale_remains_exact_before_outward_quantization():
    plan, data = _bundle()
    scale = {"incorrect": "0.123456789012345", "correct": "0.923456789012345"}
    for rating in plan["scale"]["ratings"]:
        rating["value"] = scale[rating["label"]]
    digest = measurement_plan_digest(plan)
    data["plan_sha256"] = digest
    for trial in data["trials"]:
        trial["plan_sha256"] = digest
        trial["trial_id"] = expected_trial_id(
            digest, trial["case_id"], trial["side"], trial["repetition"]
        )
        trial["parse"]["value"] = scale[trial["parse"]["rating"]]
    _retain(data)
    expected = _analyze(plan, data, _policy())
    with localcontext() as ctx:
        ctx.prec = 2
        ctx.traps[Inexact] = True
        actual = _analyze(plan, data, _policy())
    assert actual == expected
    assert actual.effect_interval.mean == D("0.8")
    assert actual.subject_interval.mean == D("0.923456789012345")
