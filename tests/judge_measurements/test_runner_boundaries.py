from __future__ import annotations

import asyncio
import copy
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.filesystem.paths import pinned_directory
from invarlock.judge_measurements import (
    CollectionOptions,
    InspectJudgeError,
    RunnerOptions,
    collect,
    render_request,
)
from invarlock.judge_measurements import runner as live
from invarlock.judge_measurements.contracts import canonical_payload

FIXTURES = Path(__file__).with_name("fixtures")


class Config(SimpleNamespace):
    def model_dump(self, *, exclude_none=False):
        return {
            key: value
            for key, value in vars(self).items()
            if not exclude_none or value is not None
        }


@pytest.fixture
def inputs():
    documents = {
        name: json.loads((FIXTURES / f"{name}.json").read_text())
        for name in ("plan", "export", "baseline_run", "subject_run")
    }
    documents["options"] = CollectionOptions.from_mapping(
        documents["export"]["collection"]
    )
    return documents


@pytest.fixture
def runner_options(tmp_path):
    return RunnerOptions(tmp_path / "checkpoint", "correctness", 30)


def _set_attribute(value, path, replacement):
    for key in path[:-1]:
        value = value[key] if isinstance(key, int) else getattr(value, key)
    key = path[-1]
    if isinstance(key, int):
        value[key] = replacement
    else:
        setattr(value, key, replacement)


@pytest.fixture
def sdk_event(inputs):
    event = copy.deepcopy(inputs["export"]["samples"][0]["events"][0])
    output = event.pop("output")
    return SimpleNamespace(
        **{
            **event,
            "input": [SimpleNamespace(**message) for message in event["input"]],
            "config": Config(**event["config"]),
            "call": SimpleNamespace(**event["call"]),
        },
        output=SimpleNamespace(
            model=output["model"],
            completion=output["completion"],
            metadata={"request_id": output["request_id"]},
            error=None,
            choices=[SimpleNamespace(stop_reason=output["finish_reason"])],
            usage=SimpleNamespace(**output["usage"]),
        ),
    )


def _project(inputs, event):
    record = inputs["baseline_run"]["records"][0]
    request = render_request(
        inputs["plan"], input_text=record["input"], answer=record["output"]
    )
    return live._project_event(
        event,
        expected_request=request,
        options=inputs["options"],
        failure_status=None,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_directory", "checkpoint", "must be a Path"),
        ("scorer_id", "", "bounded identifier"),
        ("scorer_id", "grader/other", "bounded identifier"),
        ("scorer_id", "x" * 129, "bounded identifier"),
        ("invocation_timeout_seconds", True, "between 1 and 604800"),
        ("invocation_timeout_seconds", 0, "between 1 and 604800"),
        ("invocation_timeout_seconds", 604801, "between 1 and 604800"),
    ],
)
def test_runner_options_reject_unsafe_identity_and_unbounded_deadlines(
    runner_options, field, value, message
):
    with pytest.raises(InspectJudgeError, match=message):
        replace(runner_options, **{field: value}).validate()
    assert not runner_options.checkpoint_directory.exists()


@pytest.mark.parametrize("phase", ["pending", "complete"])
def test_event_sink_preserves_first_event_when_a_second_is_emitted(phase):
    sink = live._EventSink()
    first, second = object(), object()
    emit = getattr(sink, f"on_{phase}")
    emit(first)
    with pytest.raises(InspectJudgeError, match="more than one model event"):
        emit(second)
    assert getattr(sink, phase) is first


def test_checkpoint_cannot_resume_with_a_different_scorer(inputs, runner_options):
    with pinned_directory(runner_options.checkpoint_directory, create=True) as fd:
        live._initialize_checkpoint(
            inputs["plan"], inputs["options"], runner_options, fd
        )
        header = runner_options.checkpoint_directory / "collection.json"
        original = header.read_bytes()
        with pytest.raises(InspectJudgeError, match="checkpoint identity differs"):
            live._initialize_checkpoint(
                inputs["plan"],
                inputs["options"],
                replace(runner_options, scorer_id="different-rubric"),
                fd,
            )
        assert header.read_bytes() == original


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("unknown_file", "unknown entry"),
        ("wrong_shape", "unsupported shape"),
        ("unknown_trial", "unknown identity"),
        ("boolean_attempt", "unknown identity"),
        ("duplicate_admission", "duplicate admissions"),
        ("duplicate_result", "duplicate results"),
        ("unadmitted_result", "no prior call admission"),
        ("missing_attempt", "missing, duplicated, or reordered"),
    ],
)
def test_checkpoint_replay_rejects_ambiguous_attempt_history(
    inputs, runner_options, corruption, message
):
    sample = inputs["export"]["samples"][0]
    record = {"trial_id": sample["id"], "attempt": 1, "event": sample["events"][0]}
    files = {"admission-one.json": record}
    if corruption == "unknown_file":
        files = {"unexpected.json": record}
    elif corruption == "wrong_shape":
        files["admission-one.json"] = {"trial_id": sample["id"]}
    elif corruption == "unknown_trial":
        record["trial_id"] = "unplanned-trial"
    elif corruption == "boolean_attempt":
        record["attempt"] = True
    elif corruption == "duplicate_admission":
        files["admission-two.json"] = record
    elif corruption == "duplicate_result":
        files.update({"result-one.json": record, "result-two.json": record})
    elif corruption == "unadmitted_result":
        files = {"result-one.json": record}
    elif corruption == "missing_attempt":
        record["attempt"] = 2

    with pinned_directory(runner_options.checkpoint_directory, create=True) as fd:
        for name, value in files.items():
            (runner_options.checkpoint_directory / name).write_bytes(
                canonical_payload(value)
            )
        before = {
            path.name: path.read_bytes()
            for path in runner_options.checkpoint_directory.iterdir()
        }
        with pytest.raises(InspectJudgeError, match=message):
            live._checkpoint_export(
                inputs["plan"], inputs["options"], runner_options, fd
            )
        assert {
            path.name: path.read_bytes()
            for path in runner_options.checkpoint_directory.iterdir()
        } == before


def test_collection_lock_rejects_shared_permissions_and_can_recover(runner_options):
    with pinned_directory(runner_options.checkpoint_directory, create=True) as fd:
        lock = runner_options.checkpoint_directory / ".collection.lock"
        lock.touch(mode=0o644)
        lock.chmod(0o644)
        with pytest.raises(InspectJudgeError, match="safe regular file"):
            with live._collection_lock(fd):
                pytest.fail("unsafe lock was accepted")
        lock.chmod(0o600)
        with live._collection_lock(fd) as descriptor:
            os.fstat(descriptor)
        with pytest.raises(OSError):
            os.fstat(descriptor)


class NoCallModel:
    def __init__(self):
        self.name = "example-judge"
        self.config = Config()
        self.api = SimpleNamespace(
            client=SimpleNamespace(max_retries=0),
            responses_api=False,
            model_args={},
        )
        self.model_args = {}
        self.calls = 0

    def __str__(self):
        return self.name

    async def generate(self, **kwargs):
        self.calls += 1
        pytest.fail("invalid collection configuration dispatched a provider call")


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        ("complete", "complete"),
        ("capacity", "capacity_exhausted"),
        ("before_deadline", "deadline"),
        ("during_deadline", "deadline"),
    ],
)
def test_live_stop_reason_reports_actual_admission_boundary(
    inputs, runner_options, monkeypatch, stop, expected
):
    monkeypatch.setattr(live.importlib.metadata, "version", lambda _: "0.3.263")
    monkeypatch.setattr(live, "prepare_inspect_config", lambda *_: None)
    calls = []

    async def call_one(*_args, request, **_kwargs):
        calls.append(request)
        if stop == "during_deadline":
            await asyncio.Future()
        event = copy.deepcopy(inputs["export"]["samples"][0]["events"][0])
        event["uuid"] = f"event-{len(calls)}"
        event["output"]["request_id"] = f"response-{len(calls)}"
        event["input"] = request["messages"]
        event["call"]["request"]["messages"] = request["messages"]
        return event

    monkeypatch.setattr(live, "_call_one", call_one)
    if stop == "capacity":
        from invarlock.judge_measurements import collector as planning

        # Actual live storage accounting can be stricter than the public batch
        # planner. The runner must report its own terminal admission boundary.
        monkeypatch.setattr(planning, "MAX_SOURCES", 2)
        assert not planning.prepare_collection(inputs["plan"], inputs["options"])[
            "budget_exhausted"
        ]
    elif stop == "before_deadline":
        times = iter((0, 2))
        monkeypatch.setattr(
            live, "time", SimpleNamespace(monotonic=lambda: next(times))
        )
    stops = []
    result = asyncio.run(
        collect(
            plan=inputs["plan"],
            options=inputs["options"],
            runner=replace(runner_options, invocation_timeout_seconds=1),
            model=NoCallModel(),
            baseline_run=inputs["baseline_run"],
            subject_run=inputs["subject_run"],
            on_stop=stops.append,
        )
    )
    assert stops == [expected]
    assert result["completeness"]["completed_trials"] == (
        2 if stop == "complete" else 0
    )
    if stop in {"capacity", "before_deadline"}:
        assert calls == []
        assert all(not trial["attempts"] for trial in result["trials"])


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("config",), None, "configuration is unavailable"),
        (("config", "temperature"), 0, "inherited settings"),
        (("api", "responses_api"), True, "chat-completion API"),
        (("api", "service_tier"), "priority", "service tier"),
        (("api", "service_tier"), "auto", "service tier"),
        (("model_args",), {"service_tier": "flex"}, "model_args"),
        (("model_args",), {"base_url": "https://example.com"}, "model_args"),
        (("api", "model_args"), {"max_retries": 2}, "model_args"),
        (("api", "model_args"), [], "model_args"),
        (("name",), "unapproved-judge", "model identity differs"),
    ],
)
def test_collection_rejects_inherited_model_settings_before_admission(
    inputs, runner_options, monkeypatch, path, value, message
):
    monkeypatch.setattr(live.importlib.metadata, "version", lambda _: "0.3.263")
    model = NoCallModel()
    _set_attribute(model, path, value)
    with pytest.raises(InspectJudgeError, match=message):
        asyncio.run(
            collect(
                plan=inputs["plan"],
                options=inputs["options"],
                runner=runner_options,
                model=model,
                baseline_run=inputs["baseline_run"],
                subject_run=inputs["subject_run"],
            )
        )
    assert model.calls == 0
    assert list(runner_options.checkpoint_directory.iterdir()) == []


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ("historical", "requires current Inspect version"),
        ("installed_version", "unsupported installed Inspect version"),
        ("retries", "requires max_attempts=1"),
    ],
)
def test_live_collection_rejects_unsupported_execution_profile(
    inputs, runner_options, monkeypatch, profile, message
):
    monkeypatch.setattr(
        live.importlib.metadata,
        "version",
        lambda _: "0.3.999" if profile == "installed_version" else "0.3.263",
    )
    options = inputs["options"]
    if profile == "historical":
        options = replace(options, inspect_version="0.3.254")
    elif profile == "retries":
        inputs["plan"]["schedule"].update(max_attempts=2, retry_on=["transport_error"])
    model = NoCallModel()
    with pytest.raises(InspectJudgeError, match=message):
        asyncio.run(
            collect(
                plan=inputs["plan"],
                options=options,
                runner=runner_options,
                model=model,
                baseline_run=inputs["baseline_run"],
                subject_run=inputs["subject_run"],
            )
        )
    assert model.calls == 0
    assert not list(runner_options.checkpoint_directory.glob("*.json"))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("input", 0, "content"), [{"image": "unsupported"}], "plain text"),
        (("input", 0, "role"), "tool", "plain text"),
        (("input",), None, "missing its input"),
        (("input", 0, "content"), "Unapproved prompt", "approved request"),
        (("model",), "other-judge", "approved grader"),
        (("tools",), [{"name": "search"}], "used tools"),
        (("tool_choice",), "auto", "unsupported tool choice"),
        (("retries",), 1, "SDK retries"),
        (("cache",), "reused", "used a cache"),
        (("call", "request"), None, "retain the provider request"),
        (("call", "response"), [], "response must be a JSON object"),
        (("output", "usage", "total_cost"), 0.0002, "reserved per-call amount"),
        (("output", "usage", "total_cost"), 0.0001001, "reserved per-call amount"),
        (("output", "usage", "total_cost"), float("nan"), "finite nonnegative"),
        (("output", "usage", "total_cost"), -1, "finite nonnegative"),
        (("output", "usage", "input_tokens"), 1.9, "nonnegative integers"),
        (("output", "usage", "output_tokens"), True, "nonnegative integers"),
        (("output", "usage", "input_tokens_cache_read"), -1, "nonnegative integers"),
        (
            ("output", "choices"),
            [SimpleNamespace(), SimpleNamespace()],
            "one completion",
        ),
        (
            ("output", "choices", 0, "message"),
            SimpleNamespace(tool_calls=["unrequested"]),
            "contains tool calls",
        ),
        (("config", "temperature"), 0.5, "changed generation settings"),
        (("config", "reasoning_effort"), "high", "changed generation settings"),
        (("config", "top_k"), 10, "hidden generation settings"),
    ],
)
def test_sdk_projection_rejects_unapproved_calls(
    inputs, sdk_event, path, value, message
):
    _set_attribute(sdk_event, path, value)
    with pytest.raises(InspectJudgeError, match=message):
        _project(inputs, sdk_event)


@pytest.mark.parametrize(
    "changed", ["model", "temperature", "reasoning_effort", "service_tier"]
)
def test_openai_wire_controls_are_checked_before_normalization(
    inputs, sdk_event, changed
):
    _provider_wire(inputs, sdk_event, "openai")
    if changed == "service_tier":
        sdk_event.call.response["service_tier"] = "priority"
    else:
        sdk_event.call.request[changed] = {
            "model": "changed-model",
            "temperature": 0.5,
            "reasoning_effort": "high",
        }[changed]
    with pytest.raises(InspectJudgeError, match="differs|service tier"):
        _project(inputs, sdk_event)


def test_sdk_projection_counts_cached_input_tokens_in_the_reservation(
    inputs, sdk_event
):
    sdk_event.output.usage.input_tokens_cache_read = 7
    sdk_event.output.usage.input_tokens_cache_write = 3
    projected = _project(inputs, sdk_event)
    assert projected["output"]["usage"] == {
        "input_tokens": sdk_event.output.usage.input_tokens + 10,
        "output_tokens": sdk_event.output.usage.output_tokens,
    }
    assert projected["output"]["request_id"] == sdk_event.output.metadata["request_id"]
    assert projected["call"]["response"] == {
        "format": "invarlock/judge-provider-response-v1",
        "content": sdk_event.output.completion,
        "model": sdk_event.output.model,
        "id": sdk_event.output.metadata["request_id"],
        "finish_reason": sdk_event.output.choices[0].stop_reason,
        "usage": projected["output"]["usage"],
    }
    assert projected["error"] is None


@pytest.mark.parametrize("failure", ["provider_refusal", "event_error"])
def test_failed_sdk_projection_drops_private_error_details(inputs, sdk_event, failure):
    secret = "private-provider-error"
    sdk_event.call.response = {"error": secret}
    sdk_event.output.usage = None
    sdk_event.output.choices = []
    if failure == "provider_refusal":
        sdk_event.output.error = secret
    else:
        sdk_event.error = secret
    projected = _project(inputs, sdk_event)
    assert projected["error"]["status"] == (
        "refusal" if failure == "provider_refusal" else "timeout_ambiguous"
    )
    assert projected["call"]["response"] is None
    assert projected["output"]["usage"] is None
    assert projected["output"]["completion"] == ""
    assert secret.encode() not in canonical_payload(projected)


@pytest.mark.parametrize(
    ("response", "metadata", "expected"),
    [
        ({"id": "native-id", "request_id": "fallback"}, None, "native-id"),
        ({"id": "", "request_id": "fallback"}, None, "fallback"),
        ({"response_id": "google-id"}, None, "google-id"),
        ({"responseId": "google-camel-id"}, None, "google-camel-id"),
        ({"id": "x" * 257}, "metadata-id", "metadata-id"),
        (None, {"request_id": "metadata-id"}, "metadata-id"),
        (None, {"request_id": ""}, None),
        (None, {"request_id": 17}, None),
        (None, {"request_id": "x" * 257}, None),
    ],
)
def test_request_id_uses_only_bounded_available_metadata(response, metadata, expected):
    assert live._request_id(response, SimpleNamespace(metadata=metadata)) == expected


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("record_type", "invalid record"),
        ("record_id", "invalid record"),
        ("paired_input", "paired inputs differ"),
        ("input_type", "require text inputs and outputs"),
        ("output_type", "require text inputs and outputs"),
    ],
)
def test_frozen_rows_reject_ungradeable_pairs(inputs, corruption, message):
    baseline = inputs["baseline_run"]
    subject = inputs["subject_run"]
    if corruption == "record_type":
        baseline["records"][0] = "not a record"
    elif corruption == "record_id":
        baseline["records"][0]["id"] = None
    elif corruption == "paired_input":
        subject["records"][0]["input"] = "Different task"
    elif corruption == "input_type":
        baseline["records"][0]["input"] = ["not text"]
    else:
        subject["records"][0]["output"] = None
    with pytest.raises(InspectJudgeError, match=message):
        live._frozen_rows(baseline, subject)


@pytest.mark.parametrize("failure", ["body", "unlock"])
def test_collection_lock_closes_on_body_and_unlock_failures(
    runner_options, monkeypatch, failure
):
    original_flock = live.fcntl.flock

    def flock(descriptor, operation):
        if failure == "unlock" and operation == live.fcntl.LOCK_UN:
            raise OSError("unlock failed")
        return original_flock(descriptor, operation)

    monkeypatch.setattr(live.fcntl, "flock", flock)
    with pinned_directory(runner_options.checkpoint_directory, create=True) as fd:
        with pytest.raises(OSError, match=failure):
            with live._collection_lock(fd) as descriptor:
                if failure == "body":
                    raise OSError("body failed")
        with pytest.raises(OSError):
            os.fstat(descriptor)


def _provider_wire(inputs, event, provider):
    """Start from a consistent completed call before corrupting provider evidence."""
    grader = f"{provider}/approved-model"
    inputs["plan"]["judge"]["requested_model"] = grader
    inputs["options"] = replace(inputs["options"], grader=grader)
    event.model = grader
    identity = {
        "model": event.output.model,
        "id": event.output.metadata["request_id"],
    }
    if provider == "anthropic":
        response = {
            **identity,
            "role": "assistant",
            "content": [{"type": "text", "text": event.output.completion}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": event.output.usage.input_tokens,
                "output_tokens": event.output.usage.output_tokens,
            },
        }
    elif provider == "google":
        response = {
            "modelVersion": identity["model"],
            "responseId": identity["id"],
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": event.output.completion}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": event.output.usage.input_tokens,
                "candidatesTokenCount": event.output.usage.output_tokens,
            },
        }
    else:
        response = {
            **identity,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": event.output.completion,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": event.output.usage.input_tokens,
                "completion_tokens": event.output.usage.output_tokens,
            },
        }
    if provider == "openai":
        event.call.request = {
            "model": "approved-model",
            "messages": [vars(message).copy() for message in event.input],
            "temperature": event.config.temperature,
            "top_p": event.config.top_p,
            "max_tokens": event.config.max_tokens,
        }
    event.call.response = response
    assert _project(inputs, event)["output"]["completion"] == event.output.completion
    return response


def _replace_wire_field(document, path, value):
    for key in path[:-1]:
        document = document[key]
    document[path[-1]] = value


@pytest.mark.parametrize(
    "provider,path,value,message",
    [
        ("anthropic", ("role",), "user", "unsupported shape"),
        ("anthropic", ("content",), {}, "unsupported shape"),
        ("anthropic", ("usage", "iterations"), [{"input_tokens": 1}], "iterations"),
        ("anthropic", ("content", 0), {"type": "tool_use"}, "unsupported content"),
        ("anthropic", ("content", 0, "text"), 7, "text is invalid"),
        ("anthropic", ("stop_reason",), "pause_turn", "finish reason"),
        ("anthropic", ("usage", "input_tokens"), True, "nonnegative integers"),
        ("anthropic", ("usage", "output_tokens"), -1, "nonnegative integers"),
        ("google", ("candidates",), [], "unsupported shape"),
        ("google", ("candidates",), [{}, {}], "unsupported shape"),
        ("google", ("usageMetadata",), None, "unsupported shape"),
        ("google", ("candidates", 0, "content", "role"), "user", "content is invalid"),
        ("google", ("candidates", 0, "content", "parts"), "text", "content is invalid"),
        (
            "google",
            ("candidates", 0, "content", "parts", 0),
            {"functionCall": {}},
            "unsupported content",
        ),
        (
            "google",
            ("candidates", 0, "content", "parts", 0, "text"),
            7,
            "text is invalid",
        ),
        (
            "google",
            ("candidates", 0, "finishReason"),
            "MALFORMED_FUNCTION_CALL",
            "finish reason",
        ),
        ("google", ("usageMetadata", "promptTokenCount"), "10", "nonnegative integers"),
        ("openrouter", ("choices",), None, "missing its completion"),
        ("openrouter", ("usage",), None, "missing its completion"),
    ],
)
def test_completed_call_rejects_malformed_provider_evidence(
    inputs, sdk_event, provider, path, value, message
):
    response = _provider_wire(inputs, sdk_event, provider)
    _replace_wire_field(response, path, value)
    with pytest.raises(InspectJudgeError, match=message):
        _project(inputs, sdk_event)


@pytest.mark.parametrize("provider", ["anthropic", "google"])
def test_private_reasoning_is_excluded_from_retained_judgment(
    inputs, sdk_event, provider
):
    response = _provider_wire(inputs, sdk_event, provider)
    if provider == "anthropic":
        response["content"][:0] = [
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "redacted_thinking", "data": "private signature"},
        ]
    else:
        response["candidates"][0]["content"]["parts"][:0] = [
            {"text": "private reasoning", "thought": True},
            {"thoughtSignature": "private signature"},
        ]
    projected = _project(inputs, sdk_event)
    assert projected["output"]["completion"] == sdk_event.output.completion
    assert b"private" not in canonical_payload(projected)


def test_unparseable_provider_cost_is_rejected_before_retention(inputs, sdk_event):
    sdk_event.output.usage.total_cost = "not-a-number"
    with pytest.raises(InspectJudgeError, match="finite nonnegative"):
        _project(inputs, sdk_event)


def test_anthropic_caller_cannot_bypass_continuation_reservation_guard():
    class Model(NoCallModel):
        def __str__(self):
            return "anthropic/claude-sonnet-4-5"

    model = Model()
    with pytest.raises(InspectJudgeError, match="configured single-attempt"):
        live._require_provider_retries_disabled(model)
    assert model.calls == 0


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1", 1_000_001])
def test_graceful_batch_limit_rejects_invalid_values(runner_options, value):
    with pytest.raises(InspectJudgeError, match="stop_after_batches"):
        replace(runner_options, stop_after_batches=value).validate()
    assert not runner_options.checkpoint_directory.exists()


@pytest.mark.parametrize(
    ("budget", "resume_limit", "expected_calls", "expected_stop"),
    [
        ({}, None, 4, "complete"),
        ({}, 1, 4, "complete"),
        ({"max_calls": 3}, None, 3, "capacity_exhausted"),
        ({"max_cost_microusd": 300}, None, 3, "capacity_exhausted"),
        ({"max_input_tokens": 300}, None, 3, "capacity_exhausted"),
        ({"max_output_tokens": 384}, None, 3, "capacity_exhausted"),
    ],
)
def test_graceful_stop_retains_entire_batch_and_resumes_original_budget(
    inputs,
    runner_options,
    monkeypatch,
    budget,
    resume_limit,
    expected_calls,
    expected_stop,
):
    monkeypatch.setattr(live.importlib.metadata, "version", lambda _: "0.3.263")
    monkeypatch.setattr(live, "prepare_inspect_config", lambda *_: None)
    plan = copy.deepcopy(inputs["plan"])
    plan["schedule"].update(repetitions=2, expected_trials=4)
    options = replace(inputs["options"], concurrency=2, **budget)
    calls = []

    async def completed_call(*_args, request, **_kwargs):
        identity = len(calls)
        calls.append(request)
        await asyncio.sleep(0)
        event = copy.deepcopy(inputs["export"]["samples"][0]["events"][0])
        event["uuid"] = f"event-graceful-{identity}"
        event["output"]["request_id"] = f"response-graceful-{identity}"
        event["input"] = request["messages"]
        event["call"]["request"]["messages"] = request["messages"]
        return event

    monkeypatch.setattr(live, "_call_one", completed_call)
    runner = replace(runner_options, stop_after_batches=1)
    arguments = {
        "plan": plan,
        "options": options,
        "model": NoCallModel(),
        "baseline_run": inputs["baseline_run"],
        "subject_run": inputs["subject_run"],
    }
    stops = []

    def stopped(reason):
        # The callback observes both durably finished attempts and no next
        # admission, even though two eligible trials remain in the frozen plan.
        assert len(list(runner.checkpoint_directory.glob("result-*.json"))) == 2
        assert len(list(runner.checkpoint_directory.glob("admission-*.json"))) == 2
        stops.append(reason)

    first = asyncio.run(collect(**arguments, runner=runner, on_stop=stopped))
    assert stops == ["requested"]
    assert len(calls) == 2
    assert first["completeness"]["completed_trials"] == 2
    assert sum(not trial["attempts"] for trial in first["trials"]) == 2
    original = {
        path.name: path.read_bytes()
        for path in runner.checkpoint_directory.glob("*.json")
    }
    assert "stop_after_batches" not in json.loads(original["collection.json"])
    resumed_stops = []
    resumed = asyncio.run(
        collect(
            **arguments,
            runner=replace(runner, stop_after_batches=resume_limit),
            on_stop=resumed_stops.append,
        )
    )
    assert len(calls) == expected_calls
    assert resumed_stops == [expected_stop]
    assert resumed["completeness"]["completed_trials"] == expected_calls
    assert all(
        (runner.checkpoint_directory / name).read_bytes() == raw
        for name, raw in original.items()
    )
    assert all(len(trial["attempts"]) <= 1 for trial in resumed["trials"])
    replayed = asyncio.run(collect(**arguments, runner=runner))
    assert replayed == resumed
    assert len(calls) == expected_calls


@pytest.mark.parametrize(
    "max_calls,reason", [(1, "capacity_exhausted"), (10, "complete")]
)
def test_terminal_collection_reason_precedes_graceful_request(
    inputs, runner_options, monkeypatch, max_calls, reason
):
    monkeypatch.setattr(live.importlib.metadata, "version", lambda _: "0.3.263")
    monkeypatch.setattr(live, "prepare_inspect_config", lambda *_: None)
    calls = []

    async def completed_call(*_args, request, **_kwargs):
        calls.append(request)
        event = copy.deepcopy(inputs["export"]["samples"][0]["events"][0])
        event["uuid"] = f"event-terminal-{len(calls)}"
        event["output"]["request_id"] = f"response-terminal-{len(calls)}"
        event["input"] = request["messages"]
        event["call"]["request"]["messages"] = request["messages"]
        return event

    monkeypatch.setattr(live, "_call_one", completed_call)
    stops = []
    asyncio.run(
        collect(
            plan=inputs["plan"],
            options=replace(inputs["options"], max_calls=max_calls),
            runner=replace(runner_options, stop_after_batches=1),
            model=NoCallModel(),
            baseline_run=inputs["baseline_run"],
            subject_run=inputs["subject_run"],
            on_stop=stops.append,
        )
    )
    assert stops == [reason]
