"""Test bounded parsing and attempt transitions independently of source adapters."""

from __future__ import annotations

import copy
import hashlib

import pytest

from invarlock.judge_measurements import contracts as c
from tests.judge_measurements.test_contracts import (
    _measurements,
    _plan,
    _replace_source,
    _retain_trials,
    _run,
)


def validate(value, plan=None):
    c.validate_measurements(
        value,
        plan or _plan(),
        baseline_run=_run("baseline"),
        subject_run=_run("subject"),
    )


@pytest.mark.parametrize(
    "value", [{"value": object()}, {"value": float("nan")}, {"value": "\ud800"}]
)
def test_bounded_serializer_refuses_noncanonical_values(value):
    with pytest.raises(c.JudgeMeasurementContractError, match="not canonical JSON"):
        c._bounded_canonical_payload(value, 1024, "input")


def test_bounded_serializer_refuses_growth_at_exact_utf8_boundary():
    payload = c.canonical_payload({"text": "é"})
    assert c._bounded_canonical_payload({"text": "é"}, len(payload), "input") == payload
    with pytest.raises(c.JudgeMeasurementContractError, match="exceeds"):
        c._bounded_canonical_payload({"text": "é"}, len(payload) - 1, "input")


@pytest.mark.parametrize("input_text,answer_text", [(None, "answer"), ("input", {})])
def test_request_renderer_requires_text_answers_and_inputs(input_text, answer_text):
    with pytest.raises(
        c.JudgeMeasurementContractError, match="string inputs and answers"
    ):
        c.render_judge_request(_plan(), input_text=input_text, answer_text=answer_text)


@pytest.mark.parametrize("field", ["answer_bindings", "case_units"])
def test_plan_count_limits_precede_serialization(monkeypatch, field):
    plan = _plan()
    if field == "answer_bindings":
        plan[field] = [{}] * 10001
    else:
        plan["sampling"][field] = [{}] * 10001
    monkeypatch.setattr(
        c,
        "_bounded_canonical",
        lambda *args: pytest.fail("oversized plan was serialized"),
    )
    with pytest.raises(c.JudgeMeasurementContractError, match="limit"):
        c.validate_measurement_plan(plan)


@pytest.mark.parametrize("kind", ["sources", "trials", "attempts"])
def test_measurement_count_limits_precede_serialization(monkeypatch, kind):
    value = _measurements()
    if kind == "sources":
        value["sources"] = [{}] * 1001
    elif kind == "trials":
        value["trials"] = [{}] * 200001
    else:
        value["trials"][0]["attempts"] = [{}] * 4
    original = c._bounded_canonical

    def guard(value, maximum, label):
        assert label == "judge measurement plan", (
            "oversized measurements were serialized"
        )
        return original(value, maximum, label)

    monkeypatch.setattr(c, "_bounded_canonical", guard)
    with pytest.raises(c.JudgeMeasurementContractError, match="limit"):
        validate(value)


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda t: t.update(attempts=None), "attempts must be an array"),
        (lambda t: t.update(attempts=[None]), "attempts must be objects"),
        (lambda t: t["attempts"][0].update(source=None), "source mapping"),
        (lambda t: t["attempts"][0].update(usage=[]), "usage must be an object"),
    ],
)
def test_retained_trial_integer_precheck_rejects_malformed_containers(change, message):
    trial = _measurements()["trials"][0]
    change(trial)
    with pytest.raises(c.JudgeMeasurementContractError, match=message):
        c._check_trial_integer_types(trial)


def check_attempts(trial, plan=None):
    c._check_attempts(
        trial,
        plan=plan or _plan(),
        source_ids={"source-1"},
        expected_request_sha256=_plan()["answer_bindings"][0][
            "baseline_request_sha256"
        ],
    )


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda t: t["attempts"][0].update(attempt=2), "contiguous"),
        (
            lambda t: t["attempts"][0]["source"].update(source_id="unknown"),
            "unknown source",
        ),
        (
            lambda t: t["attempts"][0]["source"].update(attempt_index=1),
            "source attempt index",
        ),
        (lambda t: t["attempts"][0].update(response=None), "require a response"),
        (
            lambda t: t["attempts"][0].update(status="transport_error"),
            "require retained error",
        ),
        (
            lambda t: t["attempts"][0].update(
                status="transport_error", error={"code": "network", "message": "failed"}
            ),
            "must not retain a completed response",
        ),
        (
            lambda t: t["attempts"][0].update(status="timeout_ambiguous"),
            "require retained error",
        ),
        (lambda t: t.update(selected_attempt=None), "first completed response"),
        (lambda t: t.update(status="incomplete"), "completeness must agree"),
    ],
)
def test_attempt_transition_rejects_inconsistent_outcomes(change, message):
    trial = _measurements()["trials"][0]
    change(trial)
    with pytest.raises(c.JudgeMeasurementContractError, match=message):
        check_attempts(trial)


def test_attempt_budget_and_retry_permission_are_independent():
    trial = _measurements()["trials"][0]
    first = trial["attempts"][0]
    final = copy.deepcopy(first)
    final["attempt"] = 2
    final["source"]["attempt_index"] = 1
    first.update(
        status="transport_error",
        response=None,
        error={"code": "network", "message": "failed"},
    )
    trial["attempts"].append(final)
    with pytest.raises(c.JudgeMeasurementContractError, match="attempt limit"):
        check_attempts(trial)
    plan = _plan()
    plan["schedule"]["max_attempts"] = 2
    plan["schedule"]["retry_on"] = []
    with pytest.raises(c.JudgeMeasurementContractError, match="retry was not approved"):
        check_attempts(trial, plan)


def test_invalid_response_json_cannot_claim_a_successful_rating():
    trial = _measurements()["trials"][0]
    raw = b'{"rating":'
    trial["attempts"][0]["response"].update(
        text=raw.decode(), sha256=hashlib.sha256(raw).hexdigest()
    )
    with pytest.raises(
        c.JudgeMeasurementContractError, match="deterministic response replay"
    ):
        check_attempts(trial)
    trial.update(
        status="incomplete", parse={"status": "invalid", "rating": None, "value": None}
    )
    check_attempts(trial)


@pytest.mark.parametrize("source", [[], {"format": "unknown", "trials": []}])
def test_retained_source_requires_a_supported_object_profile(source):
    value = _measurements()
    _replace_source(value, source)
    with pytest.raises(c.JudgeMeasurementContractError, match="unsupported retained"):
        validate(value)


def test_source_identifiers_and_replayed_trial_membership_are_unique():
    value = _measurements()
    value["sources"].append(copy.deepcopy(value["sources"][0]))
    with pytest.raises(
        c.JudgeMeasurementContractError, match="source IDs must be unique"
    ):
        validate(value)
    value = _measurements()
    _replace_source(
        value, {"format": c.SOURCE_FORMAT, "trials": [value["trials"][0]] * 2}
    )
    with pytest.raises(c.JudgeMeasurementContractError, match="unique trial IDs"):
        validate(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("recorded_trials", 1),
        ("completed_trials", 1),
        ("expected_trials", 3),
        ("status", "incomplete"),
    ],
)
def test_completeness_counts_cannot_contradict_full_replayed_schedule(field, value):
    measurements = _measurements()
    measurements["completeness"][field] = value
    with pytest.raises(
        c.JudgeMeasurementContractError,
        match="count is inconsistent|status is inconsistent",
    ):
        validate(measurements)


def test_trial_plan_binding_cannot_differ_from_bundle_plan():
    value = _measurements()
    value["trials"][0]["plan_sha256"] = "0" * 64
    _retain_trials(value)
    with pytest.raises(
        c.JudgeMeasurementContractError, match="does not bind the supplied plan"
    ):
        validate(value)


def test_trial_source_position_must_match_retained_record_index():
    value = _measurements()
    value["trials"][0]["attempts"][0]["source"]["record_index"] = 1
    _retain_trials(value)
    with pytest.raises(
        c.JudgeMeasurementContractError, match="position mapping is inconsistent"
    ):
        validate(value)
