from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from invarlock_addins.inspect_judge import (
    CollectionOptions,
    InspectJudgeError,
    RunnerOptions,
    collect,
    render_request,
)
from invarlock_addins.inspect_judge import runner as live

from invarlock.filesystem.paths import pinned_directory
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
            live._acquire_collection_lock(fd)
        lock.chmod(0o600)
        descriptor = live._acquire_collection_lock(fd)
        live._release_collection_lock(descriptor)


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
    ("path", "value", "message"),
    [
        (("config",), None, "configuration is unavailable"),
        (("config", "temperature"), 0, "inherited settings"),
        (("api", "responses_api"), True, "chat-completion API"),
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
        (("config", "temperature"), 0.5, "changed generation settings"),
        (("config", "top_k"), 10, "hidden generation settings"),
    ],
)
def test_sdk_projection_rejects_unapproved_calls(
    inputs, sdk_event, path, value, message
):
    _set_attribute(sdk_event, path, value)
    with pytest.raises(InspectJudgeError, match=message):
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
    assert projected["call"]["response"] == sdk_event.call.response
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
