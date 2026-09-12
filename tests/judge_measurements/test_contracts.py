from __future__ import annotations

import copy
import hashlib
import json
import tracemalloc
from pathlib import Path
from typing import Any, cast

import pytest

from invarlock.evaluation_records.cases import case_set_digest
from invarlock.evaluation_records.io import run_digest
from invarlock.judge_measurement_types import (
    JudgeMeasurementPlan,
    JudgeMeasurements,
)
from invarlock.judge_measurements import contracts
from invarlock.judge_measurements.contracts import JudgeMeasurementContractError
from tests.judge_measurements.test_analysis import _bundle, _runs

FIXTURES = Path(__file__).parents[1] / "fixtures" / "judge_measurements"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _plan() -> JudgeMeasurementPlan:
    return cast(JudgeMeasurementPlan, _fixture("plan.json"))


def _measurements() -> JudgeMeasurements:
    return cast(JudgeMeasurements, _fixture("measurements.json"))


def _run(side: str) -> dict[str, Any]:
    return _fixture(f"{side}_run.json")


def _retain_trials(value: JudgeMeasurements) -> None:
    source = value["sources"][0]
    payload = contracts.canonical_payload(
        {"format": contracts.SOURCE_FORMAT, "trials": value["trials"]}
    )
    source["content"] = payload.decode("utf-8")
    source["byte_size"] = len(payload)
    source["sha256"] = hashlib.sha256(payload).hexdigest()


def _replace_source(value: JudgeMeasurements, payload_value: object) -> None:
    source = value["sources"][0]
    payload = contracts.canonical_payload(payload_value)
    source["content"] = payload.decode("utf-8")
    source["byte_size"] = len(payload)
    source["sha256"] = hashlib.sha256(payload).hexdigest()


def _bind_plan(value: JudgeMeasurements, plan: JudgeMeasurementPlan) -> None:
    digest = contracts.measurement_plan_digest(plan)
    value["plan_sha256"] = digest
    for trial in value["trials"]:
        trial["plan_sha256"] = digest
        trial["trial_id"] = contracts.expected_trial_id(
            digest, trial["case_id"], trial["side"], trial["repetition"]
        )
    _retain_trials(value)


def _assert_plan_error(plan: JudgeMeasurementPlan, message: str) -> None:
    with pytest.raises(JudgeMeasurementContractError, match=message):
        contracts.validate_measurement_plan(plan)


def _assert_measurement_error(
    value: JudgeMeasurements, message: str, *, retain: bool = True
) -> None:
    if retain:
        _retain_trials(value)
    with pytest.raises(JudgeMeasurementContractError, match=message):
        contracts.validate_measurements(
            value,
            _plan(),
            baseline_run=_run("baseline"),
            subject_run=_run("subject"),
        )


def test_golden_plan_and_measurements_replay() -> None:
    plan = contracts.load_measurement_plan(FIXTURES / "plan.json")
    measurements = contracts.load_measurements(
        FIXTURES / "measurements.json",
        plan=plan,
        baseline_run=_run("baseline"),
        subject_run=_run("subject"),
    )

    assert measurements["plan_sha256"] == contracts.measurement_plan_digest(plan)
    assert measurements["completeness"] == {
        "status": "complete",
        "expected_trials": 2,
        "recorded_trials": 2,
        "completed_trials": 2,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda plan: plan["rubric"].update(sha256="0" * 64), "rubric digest"),
        (
            lambda plan: plan["prompt"]["references"].extend(
                [
                    {
                        "id": "reference",
                        "text": "a",
                        "sha256": hashlib.sha256(b"a").hexdigest(),
                    },
                    {
                        "id": "reference",
                        "text": "b",
                        "sha256": hashlib.sha256(b"b").hexdigest(),
                    },
                ]
            ),
            "reference IDs",
        ),
        (
            lambda plan: plan["scale"]["ratings"][1].update(label="incorrect"),
            "rating labels",
        ),
        (
            lambda plan: plan["scale"]["ratings"][1].update(value="0"),
            "numeric values",
        ),
        (
            lambda plan: plan["prompt"]["demonstrations"].append(
                {"input": "x", "answer": "y", "rating": "unknown"}
            ),
            "demonstration rating",
        ),
        (
            lambda plan: plan["judge"]["model_identity"].update(
                weights_sha256="0" * 64
            ),
            "hosted API",
        ),
        (
            lambda plan: plan["judge"]["model_identity"].update(kind="local_weights"),
            "local judge identity",
        ),
        (
            lambda plan: plan["sampling"]["case_units"].append(
                {"case_id": "case-1", "unit_id": "unit-2"}
            ),
            "sampling case IDs",
        ),
        (
            lambda plan: plan["answer_bindings"][0].update(case_id="case-2"),
            "same case IDs",
        ),
        (
            lambda plan: plan["schedule"].update(expected_trials=4),
            "cases × two sides × repetitions",
        ),
        (
            lambda plan: plan["schedule"].update(repetitions=1.0),
            "repetitions must be an integer",
        ),
        (
            lambda plan: plan["judge"]["config"].update(max_output_tokens=128.0),
            "max_output_tokens must be an integer",
        ),
        (
            lambda plan: plan["schedule"].update(max_attempts=2, retry_on=[]),
            "multiple attempts",
        ),
    ],
)
def test_plan_semantic_rejections(mutation: Any, message: str) -> None:
    plan = _plan()
    mutation(plan)
    _assert_plan_error(plan, message)


def test_reference_text_digest_is_checked_after_unique_ids() -> None:
    plan = _plan()
    plan["prompt"]["references"].append(
        {"id": "reference", "text": "content", "sha256": "0" * 64}
    )
    _assert_plan_error(plan, "reference 'reference' digest")


def test_local_weights_plan_accepts_bound_identity() -> None:
    plan = _plan()
    plan["judge"]["model_identity"] = {
        "kind": "local_weights",
        "weights_sha256": "0" * 64,
    }
    contracts.validate_measurement_plan(plan)


def test_plan_loader_rejects_ambiguous_and_oversized_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"format":"x","format":"y"}', encoding="utf-8")
    with pytest.raises(JudgeMeasurementContractError, match="duplicate key"):
        contracts.load_measurement_plan(duplicate)

    monkeypatch.setattr(contracts, "PLAN_MAX_BYTES", 10)
    with pytest.raises(JudgeMeasurementContractError, match="size limit"):
        contracts.load_measurement_plan(FIXTURES / "plan.json")


def test_plan_rejects_prompt_expansion_before_large_request_construction() -> None:
    plan = _plan()
    reference = "r" * 100_000
    plan["prompt"]["references"] = [
        {
            "id": "large-reference",
            "text": reference,
            "sha256": hashlib.sha256(reference.encode()).hexdigest(),
        }
    ]
    plan["prompt"]["demonstrations"] = [
        {"input": f"example-{index}", "answer": "answer", "rating": "correct"}
        for index in range(20)
    ]

    _assert_plan_error(plan, "normalized judge request exceeds")


def test_trial_ids_are_stable_and_bound_to_every_slot_dimension() -> None:
    plan_digest = contracts.measurement_plan_digest(_plan())
    expected = contracts.expected_trial_id(plan_digest, "case-1", "baseline", 1)

    assert expected == _measurements()["trials"][0]["trial_id"]
    assert (
        len(
            {
                contracts.expected_trial_id(plan_digest, case, side, repetition)
                for case in ("case-1", "case-2")
                for side in ("baseline", "subject")
                for repetition in (1, 2)
            }
        )
        == 8
    )


@pytest.mark.parametrize(
    ("mutation", "message", "retain"),
    [
        (
            lambda value: value.update(plan_sha256="0" * 64),
            "do not bind the supplied plan",
            True,
        ),
        (
            lambda value: value["sources"][0].update(byte_size=1),
            "byte_size",
            False,
        ),
        (
            lambda value: value["sources"][0].update(sha256="0" * 64),
            "digest",
            False,
        ),
        (
            lambda value: value["trials"][0].update(trial_id="wrong"),
            "trial ID",
            True,
        ),
        (
            lambda value: value["trials"][0].update(answer_sha256="0" * 64),
            "frozen answer",
            True,
        ),
        (
            lambda value: value["trials"][0]["attempts"][0].update(cache="reused"),
            "forbids reused",
            True,
        ),
        (
            lambda value: value["trials"][0]["attempts"][0].update(
                resolved_model="unapproved"
            ),
            "unapproved resolved model",
            True,
        ),
        (
            lambda value: value["trials"][0]["attempts"][0]["request"].update(
                sha256="0" * 64
            ),
            "request digest",
            True,
        ),
        (
            lambda value: value["trials"][0]["parse"].update(
                rating="incorrect", value="1"
            ),
            "deterministic response replay",
            True,
        ),
        (
            lambda value: value["completeness"].update(completed_trials=1),
            "completed-trial count",
            True,
        ),
    ],
)
def test_measurement_semantic_rejections(
    mutation: Any, message: str, retain: bool
) -> None:
    measurements = _measurements()
    mutation(measurements)
    _assert_measurement_error(measurements, message, retain=retain)


def test_source_replay_rejects_outer_trial_substitution() -> None:
    measurements = _measurements()
    measurements["trials"][0]["attempts"][0]["request_id"] = "substituted"
    _assert_measurement_error(measurements, "source replay differs", retain=False)


def test_retained_source_rejects_nonobject_trials_and_attempts() -> None:
    measurements = _measurements()
    _replace_source(measurements, {"format": contracts.SOURCE_FORMAT, "trials": [None]})
    _assert_measurement_error(measurements, "trials must be objects", retain=False)

    measurements = _measurements()
    retained_trial = copy.deepcopy(measurements["trials"][0])
    retained_trial["attempts"] = [None]
    _replace_source(
        measurements,
        {"format": contracts.SOURCE_FORMAT, "trials": [retained_trial]},
    )
    _assert_measurement_error(measurements, "attempts must be objects", retain=False)


def test_integral_floats_are_rejected_before_integer_interpretation() -> None:
    measurements = _measurements()
    cast(dict[str, Any], measurements["trials"][0])["selected_attempt"] = 1.0
    _retain_trials(measurements)
    _assert_measurement_error(
        measurements, "selected attempt must be an integer", retain=False
    )


def test_one_model_event_cannot_satisfy_multiple_trials() -> None:
    measurements = _measurements()
    first_event = measurements["trials"][0]["attempts"][0]["source"]["model_event_id"]
    measurements["trials"][1]["attempts"][0]["source"]["model_event_id"] = first_event
    _assert_measurement_error(measurements, "exactly one attempt", retain=True)


def test_all_measurement_integer_fields_reject_integral_floats() -> None:
    measurements = _measurements()
    cast(dict[str, Any], measurements["sources"][0])["byte_size"] = float(
        measurements["sources"][0]["byte_size"]
    )
    _assert_measurement_error(
        measurements, "source byte_size must be an integer", retain=False
    )

    measurements = _measurements()
    usage = measurements["trials"][0]["attempts"][0]["usage"]
    assert usage is not None
    cast(dict[str, Any], usage)["input_tokens"] = 35.0
    _retain_trials(measurements)
    _assert_measurement_error(
        measurements, "usage input_tokens must be an integer", retain=False
    )


def test_frozen_run_membership_must_exactly_match_plan_cases() -> None:
    plan = _plan()
    binding = copy.deepcopy(plan["answer_bindings"][0])
    binding["case_id"] = "case-2"
    plan["answer_bindings"].append(binding)
    plan["sampling"]["case_units"].append({"case_id": "case-2", "unit_id": "unit-2"})
    plan["schedule"]["expected_trials"] = 4
    measurements = _measurements()
    _bind_plan(measurements, plan)

    with pytest.raises(JudgeMeasurementContractError, match="membership"):
        contracts.validate_measurements(
            measurements,
            plan,
            baseline_run=_run("baseline"),
            subject_run=_run("subject"),
        )


def test_extra_frozen_run_case_cannot_be_hidden_by_plan_subset() -> None:
    plan = _plan()
    runs = {side: _run(side) for side in ("baseline", "subject")}
    for run in runs.values():
        extra = copy.deepcopy(run["records"][0])
        extra["id"] = "case-extra"
        run["records"].append(extra)
    plan["baseline_run_sha256"] = run_digest(runs["baseline"])
    plan["subject_run_sha256"] = run_digest(runs["subject"])
    plan["case_set_sha256"] = case_set_digest(
        {
            "format": "invarlock/evaluation-case-set-v1",
            "cases": [
                {key: row[key] for key in ("id", "input", "expected", "metadata")}
                for row in runs["baseline"]["records"]
            ],
        }
    )
    measurements = _measurements()
    _bind_plan(measurements, plan)

    with pytest.raises(JudgeMeasurementContractError, match="membership"):
        contracts.validate_measurements(
            measurements,
            plan,
            baseline_run=runs["baseline"],
            subject_run=runs["subject"],
        )


def test_rehashed_unapproved_rendered_request_is_rejected() -> None:
    measurements = _measurements()
    request = measurements["trials"][0]["attempts"][0]["request"]
    request["text"] = '{"messages":[{"role":"user","content":"Unapproved"}]}'
    request["sha256"] = hashlib.sha256(request["text"].encode()).hexdigest()
    _assert_measurement_error(measurements, "request was not approved", retain=True)


def test_frozen_request_binding_must_match_rendered_answer() -> None:
    plan = _plan()
    plan["answer_bindings"][0]["baseline_request_sha256"] = "0" * 64
    measurements = _measurements()
    _bind_plan(measurements, plan)

    with pytest.raises(JudgeMeasurementContractError, match="request binding"):
        contracts.validate_measurements(
            measurements,
            plan,
            baseline_run=_run("baseline"),
            subject_run=_run("subject"),
        )


def test_declared_source_profile_must_match_the_retained_sources() -> None:
    measurements = _measurements()
    measurements["source_profile"] = "retained-inspect-model-events-v1"

    _assert_measurement_error(measurements, "source profile differs")


def test_response_limit_counts_utf8_bytes_instead_of_characters() -> None:
    measurements = _measurements()
    response = measurements["trials"][0]["attempts"][0]["response"]
    assert response is not None
    content = "😀" * 400_000
    response.update(
        text=content,
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    _assert_measurement_error(measurements, "judge response exceeds", retain=True)


def test_frozen_binding_validation_streams_expanded_requests() -> None:
    plan, _ = _bundle(groups=tuple(f"unit-{index}" for index in range(32)))
    baseline, subject = _runs(plan)
    plan["prompt"]["system"] = "x" * (256 * 1024)
    rows = {
        "baseline": {row["id"]: row for row in baseline["records"]},
        "subject": {row["id"]: row for row in subject["records"]},
    }
    for binding in plan["answer_bindings"]:
        for side in ("baseline", "subject"):
            row = rows[side][binding["case_id"]]
            request = contracts.render_judge_request(
                plan, input_text=row["input"], answer_text=row["output"]
            )
            binding[f"{side}_request_sha256"] = hashlib.sha256(request).hexdigest()

    tracemalloc.start()
    try:
        assert (
            contracts._validate_frozen_answer_bindings(plan, baseline, subject) is None
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 12 * 1024 * 1024


def test_incomplete_trial_is_retained_but_cannot_claim_complete() -> None:
    measurements = _measurements()
    trial = measurements["trials"][0]
    attempt = trial["attempts"][0]
    attempt.update(
        status="timeout_ambiguous",
        resolved_model=None,
        response=None,
        request_id=None,
        finish_reason=None,
        error={"code": "timeout", "message": "Completion state is unknown."},
        usage=None,
    )
    trial.update(
        status="incomplete",
        selected_attempt=None,
        parse={"status": "unavailable", "rating": None, "value": None},
    )
    measurements["completeness"].update(status="incomplete", completed_trials=1)
    _retain_trials(measurements)

    contracts.validate_measurements(
        measurements,
        _plan(),
        baseline_run=_run("baseline"),
        subject_run=_run("subject"),
    )


def test_transport_retry_must_be_contiguous_and_stop_at_first_completion() -> None:
    measurements = _measurements()
    plan = _plan()
    plan["schedule"]["max_attempts"] = 2
    first = measurements["trials"][0]["attempts"][0]
    duplicate = copy.deepcopy(first)
    duplicate["attempt"] = 2
    duplicate["source"]["attempt_index"] = 1
    measurements["trials"][0]["attempts"].append(duplicate)
    _bind_plan(measurements, plan)

    with pytest.raises(JudgeMeasurementContractError, match="after a terminal result"):
        contracts.validate_measurements(
            measurements,
            plan,
            baseline_run=_run("baseline"),
            subject_run=_run("subject"),
        )


def test_transport_retry_then_first_completion_is_valid() -> None:
    measurements = _measurements()
    plan = _plan()
    plan["schedule"]["max_attempts"] = 2
    first = measurements["trials"][0]["attempts"][0]
    completed = copy.deepcopy(first)
    completed["attempt"] = 2
    completed["request_id"] = "request-after-retry"
    completed["source"]["attempt_index"] = 1
    completed["source"]["model_event_id"] = "event-after-retry"
    first.update(
        status="transport_error",
        resolved_model=None,
        response=None,
        request_id=None,
        finish_reason=None,
        error={"code": "transport", "message": "Connection failed."},
        usage=None,
    )
    measurements["trials"][0]["attempts"].append(completed)
    measurements["trials"][0]["selected_attempt"] = 2
    _bind_plan(measurements, plan)

    contracts.validate_measurements(
        measurements,
        plan,
        baseline_run=_run("baseline"),
        subject_run=_run("subject"),
    )


def test_missing_or_extra_source_trials_never_become_survivor_analysis() -> None:
    measurements = _measurements()
    measurements["trials"].pop()
    measurements["completeness"].update(
        status="incomplete", recorded_trials=1, completed_trials=1
    )
    _retain_trials(measurements)
    _assert_measurement_error(measurements, "contract is invalid", retain=False)


def test_schema_diagnostics_are_bounded() -> None:
    plan = cast(JudgeMeasurementPlan, _fixture("plan.json"))
    cast(dict[str, Any], plan)["unexpected"] = "x" * 1000
    with pytest.raises(JudgeMeasurementContractError) as caught:
        contracts.validate_measurement_plan(plan)
    assert len(str(caught.value)) < 400
