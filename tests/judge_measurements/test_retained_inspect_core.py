"""Exercise offline Inspect replay without importing the optional collector SDK."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from invarlock.judge_measurements import contracts as c
from tests.judge_measurements.test_contracts import _measurements, _plan, _run


@pytest.fixture
def retained():
    exported = json.loads(
        (
            Path(__file__).parents[2]
            / "examples/judge-measurements/inspect-export.json"
        ).read_text()
    )
    measurements = _measurements()
    measurements["source_profile"] = "retained-inspect-model-events-v1"
    measurements["sources"][0]["profile"] = measurements["source_profile"]
    records = []
    for trial, sample in zip(measurements["trials"], exported["samples"], strict=True):
        event = copy.deepcopy(sample["events"][0])
        attempt = trial["attempts"][0]
        event["uuid"] = attempt["source"]["model_event_id"]
        event["output"].update(
            model=attempt["resolved_model"],
            request_id=attempt["request_id"],
            usage=copy.deepcopy(attempt["usage"]),
        )
        records.append({"trial": trial, "events": [event]})
    source = {
        "format": c.INSPECT_SOURCE_FORMAT,
        "inspect_version": exported["inspect_version"],
        "collection": exported["collection"],
        "records": records,
    }
    return measurements, source


def replay(retained):
    measurements, source = retained
    payload = c.canonical_payload(source)
    measurements["sources"][0].update(
        content=payload.decode(),
        byte_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    c.validate_measurements(
        measurements,
        _plan(),
        baseline_run=_run("baseline"),
        subject_run=_run("subject"),
    )


def event_parts(retained):
    source = retained[1]
    record = source["records"][0]
    return record["trial"]["attempts"][0], record["events"][0], source["collection"]


def set_path(value, path, replacement):
    keys = path.split(".")
    for key in keys[:-1]:
        value = value[int(key)] if isinstance(value, list) else value[key]
    key = int(keys[-1]) if isinstance(value, list) else keys[-1]
    value[key] = replacement


def native(event):
    normalized = event["call"]["request"]
    config = normalized["config"]
    event["call"]["request"] = {
        "model": normalized["model"],
        "messages": copy.deepcopy(normalized["messages"]),
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "max_tokens": config["max_output_tokens"],
        "seed": config["seed"],
        "n": 1,
    }
    output = event["output"]
    event["call"]["response"] = {
        "model": output["model"],
        "id": output["request_id"],
        "choices": [
            {"message": {"content": output["completion"]}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": output["usage"]["input_tokens"],
            "completion_tokens": output["usage"]["output_tokens"],
        },
    }
    return event["call"]


def test_retained_inspect_normalized_and_native_events_replay_in_core(retained):
    replay(retained)
    for record in retained[1]["records"]:
        call = native(record["events"][0])
        call["request"].update(
            tools=None, tool_choice=None, extra_headers={"x-irid": "correlation"}
        )
    replay(retained)


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("extra", True, "unsupported fields"),
        ("uuid", "other-event", "does not match"),
        ("role", "solver", "does not match"),
        ("output", {}, "unsupported fields"),
        ("cache", "cache-key", "cache or SDK retries"),
        ("retries", 1, "cache or SDK retries"),
        ("tools", ["tool"], "cannot use tools"),
        ("tool_choice", "auto", "cannot use tools"),
        ("input", [], "input differs"),
        ("model", "other-model", "input differs"),
        ("config.temperature", 1, "generation config differs"),
        ("call", {"request": {}}, "call is incomplete"),
        ("call.request", [], "call is incomplete"),
        ("output.model", "different", "resolved model does not match"),
        ("output.usage.input_tokens", 34, "usage does not match"),
        ("call.error", True, "inconsistent outcome"),
        ("error", {"status": "cancelled"}, "inconsistent outcome"),
        ("output.request_id", "different", "metadata does not match"),
        ("output.finish_reason", "max_tokens", "metadata does not match"),
        ("call.request.config.top_p", "0.5", "normalized provider request differs"),
        ("call.response", {"rating": "incorrect"}, "contradicts its completion"),
        ("call.response", {"unsupported": True}, "unsupported shape"),
        ("call.response", None, "response must be an object"),
        ("output.completion", None, "completion must be text"),
        ("call.response.Authorization", "credential", "credential field"),
        ("call.request.extra_headers", {"cookie": "secret"}, "unsupported headers"),
    ],
)
def test_retained_event_contradictions_fail_before_analysis(
    retained, path, value, message
):
    attempt, event, collection = event_parts(retained)
    set_path(event, path, value)
    with pytest.raises(c.JudgeMeasurementContractError, match=message):
        c._check_retained_inspect_event(attempt, event, collection)


@pytest.mark.parametrize(
    "normalized_request,message",
    [
        ({}, "normalized request is invalid"),
        ({"text": "{"}, "normalized request is invalid"),
        ({"text": "[]"}, "input differs"),
    ],
)
def test_retained_event_requires_a_decodable_normalized_request(
    retained, normalized_request, message
):
    attempt, event, collection = event_parts(retained)
    attempt["request"] = normalized_request
    with pytest.raises(c.JudgeMeasurementContractError, match=message):
        c._check_retained_inspect_event(attempt, event, collection)


@pytest.mark.parametrize(
    "configuration",
    [
        None,
        {"temperature": True, "top_p": "1"},
        {"temperature": "invalid", "top_p": "1"},
    ],
)
def test_retained_normalized_config_cannot_hide_invalid_numbers(
    retained, configuration
):
    attempt, event, collection = event_parts(retained)
    request = json.loads(attempt["request"]["text"])
    request["config"] = configuration
    attempt["request"]["text"] = c.canonical_payload(request).decode()
    with pytest.raises(c.JudgeMeasurementContractError, match="normalized config"):
        c._check_retained_inspect_event(attempt, event, collection)


@pytest.mark.parametrize("attempt,event", [(None, {}), ({}, None)])
def test_retained_event_requires_object_inputs(attempt, event):
    with pytest.raises(c.JudgeMeasurementContractError, match="must be objects"):
        c._check_retained_inspect_event(attempt, event, {})


@pytest.mark.parametrize(
    "usage,completed,message",
    [
        (None, True, "requires token usage"),
        ({"tokens": 1}, False, "usage is invalid"),
        ({"input_tokens": 101, "output_tokens": 5}, True, "per-call reservation"),
        ({"input_tokens": 35, "output_tokens": True}, True, "per-call reservation"),
    ],
)
def test_usage_is_complete_strict_and_within_its_reservation(usage, completed, message):
    with pytest.raises(c.JudgeMeasurementContractError, match=message):
        c._check_retained_inspect_usage(
            usage,
            copy.deepcopy(usage),
            completed=completed,
            maximum_input_tokens=100,
            maximum_output_tokens=128,
        )


def failed_event(retained, status="timeout_ambiguous"):
    attempt, event, collection = event_parts(retained)
    error = {"code": "interrupted", "message": "No complete response retained"}
    attempt.update(
        status=status,
        response=None,
        resolved_model=None,
        usage=None,
        error=error,
        request_id=None,
        finish_reason=None,
    )
    event["call"].update(response=None, error=True)
    event["output"].update(
        model=None, usage=None, request_id=None, finish_reason=None, completion=""
    )
    event["error"] = {"status": status, **error}
    trial = retained[1]["records"][0]["trial"]
    trial.update(
        status="incomplete",
        selected_attempt=None,
        parse={
            "status": "refusal" if status == "refusal" else "unavailable",
            "rating": None,
            "value": None,
        },
    )
    retained[0]["completeness"].update(status="incomplete", completed_trials=1)
    return attempt, event, collection


@pytest.mark.parametrize(
    "status", ["timeout_ambiguous", "cancelled", "refusal", "transport_error"]
)
def test_failed_calls_remain_replayable_incomplete_evidence(retained, status):
    failed_event(retained, status)
    replay(retained)


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("error", None, "inconsistent outcome"),
        ("call.error", False, "inconsistent outcome"),
        ("call.response", [], "inconsistent outcome"),
        ("error.status", "cancelled", "error differs"),
        ("error.message", "different", "error differs"),
    ],
)
def test_failure_projection_cannot_discard_or_relabel_an_error(
    retained, path, value, message
):
    attempt, event, collection = failed_event(retained)
    set_path(event, path, value)
    with pytest.raises(c.JudgeMeasurementContractError, match=message):
        c._check_retained_inspect_event(attempt, event, collection)


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("extra", True, "unsupported shape"),
        ("inspect_version", "future", "unsupported profile"),
        ("collection", {}, "options must be an object"),
        ("collection.grader", "https://grader", "configuration is unsupported"),
        ("collection.epochs", True, "configuration is unsupported"),
        ("collection.max_calls", 0, "supported range"),
        ("records.0", {}, "one trial and its events"),
        ("records.0.events", [], "event count"),
    ],
)
def test_source_envelope_and_record_structure_are_closed(
    retained, path, value, message
):
    set_path(retained[1], path, value)
    with pytest.raises(c.JudgeMeasurementContractError, match=message):
        replay(retained)


@pytest.mark.parametrize(
    "budget,value",
    [
        ("max_calls", 1),
        ("max_input_tokens", 100),
        ("max_output_tokens", 128),
        ("max_cost_microusd", 100),
    ],
)
def test_retained_source_respects_each_aggregate_reservation(retained, budget, value):
    retained[1]["collection"][budget] = value
    with pytest.raises(c.JudgeMeasurementContractError, match="resource reservations"):
        replay(retained)


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("request.extra", 1, "unsupported controls"),
        ("request.messages", None, "chat messages"),
        ("request.messages.0", {"role": "user", "content": 1}, "plain text"),
        ("request.messages.0.role", "assistant", "messages differ"),
        ("request.model", "different", "model differs"),
        ("request.temperature", True, "config is incomplete"),
        ("request.temperature", 1, "config differs"),
        ("request.seed", False, "seed differs"),
        ("request.max_tokens", True, "token limit differs"),
        ("request.n", 2, "one completion"),
        ("request.tools", ["tool"], "contains tools"),
        ("request.tool_choice", "auto", "tool choice"),
        ("response.model", "wrong", "resolved model"),
        ("response.id", "wrong", "request ID"),
        ("response.choices.0.finish_reason", "tool_calls", "finish reason"),
        ("response.usage.prompt_tokens", 34, "usage contradicts"),
    ],
)
def test_native_provider_projection_binds_controls_and_response_metadata(
    retained, path, value, message
):
    attempt, event, collection = event_parts(retained)
    set_path(native(event), path, value)
    with pytest.raises(c.JudgeMeasurementContractError, match=message):
        c._check_retained_inspect_event(attempt, event, collection)


@pytest.mark.parametrize(
    "field,message",
    [("temperature", "config is incomplete"), ("max_tokens", "one token limit")],
)
def test_native_provider_required_controls_cannot_be_omitted(retained, field, message):
    attempt, event, collection = event_parts(retained)
    native(event)["request"].pop(field)
    with pytest.raises(c.JudgeMeasurementContractError, match=message):
        c._check_retained_inspect_event(attempt, event, collection)


@pytest.mark.parametrize("completion", ["not JSON", '{"rating":'])
def test_malformed_native_completion_is_retained_as_invalid_not_repaired(
    retained, completion
):
    attempt, event, _ = event_parts(retained)
    event["output"]["completion"] = completion
    native(event)
    raw = c.canonical_payload(completion)
    attempt["response"].update(
        text=raw.decode(), sha256=hashlib.sha256(raw).hexdigest()
    )
    retained[1]["records"][0]["trial"].update(
        status="incomplete", parse={"status": "invalid", "rating": None, "value": None}
    )
    retained[0]["completeness"].update(status="incomplete", completed_trials=1)
    replay(retained)


def test_normalized_completion_must_match_selected_trial_response(retained):
    attempt, event, collection = event_parts(retained)
    attempt["response"]["text"] = '{"rating":"incorrect"}'
    with pytest.raises(
        c.JudgeMeasurementContractError, match="completion does not match"
    ):
        c._check_retained_inspect_event(attempt, event, collection)


@pytest.mark.parametrize("mutation", ["role", "temperature", "token_limit"])
def test_sol_projection_is_version_bound_and_preserves_approved_semantics(
    retained, mutation
):
    attempt, event, collection = event_parts(retained)
    normalized = json.loads(attempt["request"]["text"])
    normalized["model"] = "openai/gpt-5.6-sol"
    normalized["config"]["temperature"] = "1"
    event["model"] = normalized["model"]
    event["config"]["temperature"] = 1.0
    event["call"]["request"] = normalized
    attempt["request"]["text"] = c.canonical_payload(normalized).decode()
    call = native(event)
    call["request"]["messages"][0]["role"] = "developer"
    call["request"].pop("temperature")
    call["request"]["max_completion_tokens"] = call["request"].pop("max_tokens")
    c._check_retained_inspect_event(attempt, event, collection)
    if mutation == "role":
        call["request"]["messages"][0]["role"] = "system"
    elif mutation == "temperature":
        call["request"]["temperature"] = 1
    else:
        call["request"]["max_tokens"] = call["request"].pop("max_completion_tokens")
    with pytest.raises(
        c.JudgeMeasurementContractError, match="projection|messages differ"
    ):
        c._check_retained_inspect_event(attempt, event, collection)
