"""Reject rehashed malformed answers, retained sources, and trial identities."""

import copy
import hashlib
import json

import pytest

from invarlock.evaluation_records.io import run_digest
from invarlock.judge_measurements import contracts
from tests.judge_measurements.test_contracts import (
    _assert_measurement_error,
    _bind_plan,
    _measurements,
    _plan,
    _run,
)


def test_plan_rejects_duplicate_answer_binding_even_with_different_digests():
    plan = _plan()
    duplicate = copy.deepcopy(plan["answer_bindings"][0])
    duplicate["baseline_answer_sha256"] = "0" * 64
    plan["answer_bindings"].append(duplicate)
    with pytest.raises(
        contracts.JudgeMeasurementContractError,
        match="answer-binding case IDs must be unique",
    ):
        contracts.validate_measurement_plan(plan)


def test_seeded_judge_without_system_message_renders_exact_declared_configuration():
    plan = _plan()
    plan["prompt"]["system"] = ""
    plan["judge"]["config"]["seed"] = 42
    contracts.validate_measurement_plan(plan)
    request = json.loads(
        contracts.render_judge_request(
            plan, input_text="Question", answer_text="Answer"
        )
    )
    assert request["config"]["seed"] == 42
    assert [message["role"] for message in request["messages"]] == ["user"]
    assert json.loads(request["messages"][0]["content"])["answer"] == "Answer"


@pytest.mark.parametrize(
    "record_change, message",
    [
        ({"error": "generation failed"}, "cannot grade a failed frozen answer"),
        ({"output": {"answer": "Paris"}}, "requires string inputs and answers"),
    ],
)
def test_rehashing_a_failed_or_nontext_run_cannot_make_it_judgeable(
    record_change, message
):
    plan = _plan()
    baseline, subject = _run("baseline"), _run("subject")
    baseline["records"][0].update(record_change)
    plan["baseline_run_sha256"] = run_digest(baseline)
    measurements = _measurements()
    _bind_plan(measurements, plan)
    with pytest.raises(contracts.JudgeMeasurementContractError, match=message):
        contracts.validate_measurements(
            measurements, plan, baseline_run=baseline, subject_run=subject
        )


def test_invalid_frozen_run_is_reported_as_a_measurement_contract_error():
    baseline = _run("baseline")
    del baseline["source"]
    with pytest.raises(
        contracts.JudgeMeasurementContractError, match="frozen answer run is invalid"
    ):
        contracts.validate_measurements(
            _measurements(), _plan(), baseline_run=baseline, subject_run=_run("subject")
        )


@pytest.mark.parametrize(
    "content, message",
    [
        ('{"format":', "source"),
        (
            '{ "format": "invarlock/retained-judge-json-v1", "trials": [] }',
            "must use canonical JSON",
        ),
    ],
)
def test_rehashing_retained_content_does_not_bypass_json_validation(content, message):
    measurements = _measurements()
    source = measurements["sources"][0]
    raw = content.encode()
    source.update(
        content=content, byte_size=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )
    _assert_measurement_error(measurements, message, retain=False)


def test_trial_cannot_substitute_a_different_plan_while_preserving_its_slot_id():
    measurements = _measurements()
    measurements["trials"][0]["plan_sha256"] = "0" * 64
    _assert_measurement_error(measurements, "does not bind the supplied plan")


def test_model_event_cannot_be_reused_across_distinct_trial_positions():
    measurements = _measurements()
    first, second = measurements["trials"]
    second["attempts"][0]["source"]["model_event_id"] = first["attempts"][0]["source"][
        "model_event_id"
    ]
    _assert_measurement_error(
        measurements, "model events must belong to exactly one attempt"
    )


def test_complete_trials_cannot_claim_an_incomplete_summary():
    measurements = _measurements()
    measurements["completeness"]["status"] = "incomplete"
    _assert_measurement_error(measurements, "completeness status is inconsistent")


@pytest.mark.parametrize("field, value", [("trials", None), ("trials", [None])])
def test_invalid_trial_containers_fail_with_a_schema_error(field, value):
    measurements = _measurements()
    measurements[field] = value
    _assert_measurement_error(measurements, "contract is invalid", retain=False)


def test_invalid_sampling_container_fails_with_a_schema_error():
    plan = _plan()
    plan["sampling"] = None
    with pytest.raises(
        contracts.JudgeMeasurementContractError, match="contract is invalid"
    ):
        contracts.validate_measurement_plan(plan)
