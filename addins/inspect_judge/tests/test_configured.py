"""Configured collection fails before calls and always releases its client."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from invarlock_addins.inspect_judge import (
    CollectionOptions,
    InspectJudgeError,
    RunnerOptions,
    bind_requests,
    collect_configured,
    configured,
    validate_collection_environment,
)
from invarlock_addins.inspect_judge.runner import (
    _require_clean_model_configuration,
    _require_provider_retries_disabled,
)

FIXTURES = Path(__file__).with_name("fixtures")
KEY = "offline-test-key"


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    documents = {
        name: json.loads((FIXTURES / f"{name}.json").read_text())
        for name in ("plan", "export", "frozen", "baseline_run", "subject_run")
    }
    documents["plan"]["judge"].update(
        provider="openai",
        requested_model="openai/gpt-4o-2024-08-06",
        approved_resolved_models=["gpt-4o-2024-08-06"],
    )
    return {
        "plan": bind_requests(documents["plan"], documents["frozen"]),
        "options": replace(
            CollectionOptions.from_mapping(documents["export"]["collection"]),
            grader="openai/gpt-4o-2024-08-06",
        ),
        "runner": RunnerOptions(tmp_path / "checkpoint", "correctness", 30),
        "baseline_run": documents["baseline_run"],
        "subject_run": documents["subject_run"],
    }


@pytest.fixture
def sdk(monkeypatch):
    config = SimpleNamespace(model_dump=lambda **kwargs: {})
    client = SimpleNamespace(
        max_retries=0,
        base_url="https://api.openai.com/v1/",
        close=AsyncMock(),
    )
    model = SimpleNamespace(
        config=config,
        model_args={"max_retries": 0, "responses_api": False},
        api=SimpleNamespace(client=client, responses_api=False, model_args={}),
    )
    module = SimpleNamespace(
        get_model=Mock(return_value=model), GenerateConfig=Mock(return_value=config)
    )
    monkeypatch.setattr(
        configured.importlib.metadata, "version", configured._SDK_VERSIONS.__getitem__
    )
    monkeypatch.setattr(
        configured.importlib, "import_module", Mock(return_value=module)
    )
    return module, model, client


def test_preflight_metadata_is_usable_and_never_constructs_model(inputs, sdk):
    module, _, _ = sdk
    metadata = validate_collection_environment(
        inputs["options"], {"OPENAI_API_KEY": KEY}
    )
    assert metadata == {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "sdk_versions": {
            "inspect-ai": "0.3.263",
            "openai": "3.13.0",
            "httpx": "0.28.1",
        },
        "credential_available": True,
    }
    assert KEY not in json.dumps(metadata)
    assert not module.get_model.called
    metadata["sdk_versions"]["openai"] = "changed"
    assert configured._SDK_VERSIONS["openai"] == "3.13.0"
    assert not validate_collection_environment(inputs["options"], {}, False)[
        "credential_available"
    ]


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_key_fails_before_model_construction(inputs, sdk, key):
    environment = {} if key is None else {"OPENAI_API_KEY": key}
    with pytest.raises(InspectJudgeError, match="OPENAI_API_KEY"):
        asyncio.run(collect_configured(**inputs, environment=environment))
    sdk[0].get_model.assert_not_called()
    assert not inputs["runner"].checkpoint_directory.exists()


@pytest.mark.parametrize("name", ["OPENAI_BASE_URL", "OPENAI_API_BASE"])
@pytest.mark.parametrize(
    "value", ["", "https://api.openai.com/v1", "https://example.invalid/private"]
)
@pytest.mark.parametrize("inherited", [False, True])
def test_endpoint_overrides_rejected_without_echoing_values(
    inputs, sdk, monkeypatch, name, value, inherited
):
    environment = {"OPENAI_API_KEY": KEY}
    if inherited:
        monkeypatch.setenv(name, value)
    else:
        environment[name] = value
    with pytest.raises(
        InspectJudgeError, match="endpoint environment overrides"
    ) as error:
        asyncio.run(collect_configured(**inputs, environment=environment))
    assert KEY not in str(error.value)
    assert "example.invalid" not in str(error.value)
    sdk[0].get_model.assert_not_called()


@pytest.mark.parametrize("distribution", ["inspect-ai", "openai", "httpx"])
@pytest.mark.parametrize("missing", [False, True])
def test_exact_pins_and_missing_extra_fail_before_model(
    inputs, sdk, monkeypatch, distribution, missing
):
    def version(name):
        if name == distribution:
            if missing:
                raise importlib.metadata.PackageNotFoundError(KEY)
            return "0.0.0"
        return configured._SDK_VERSIONS[name]

    monkeypatch.setattr(configured.importlib.metadata, "version", version)
    with pytest.raises(InspectJudgeError, match="install|requires") as error:
        asyncio.run(collect_configured(**inputs, environment={"OPENAI_API_KEY": KEY}))
    assert KEY not in str(error.value)
    sdk[0].get_model.assert_not_called()


def test_import_failure_is_sanitized_before_model_construction(
    inputs, sdk, monkeypatch
):
    monkeypatch.setattr(
        configured.importlib, "import_module", Mock(side_effect=RuntimeError(KEY))
    )
    with pytest.raises(InspectJudgeError, match="could not be imported") as error:
        validate_collection_environment(inputs["options"], {"OPENAI_API_KEY": KEY})
    assert KEY not in str(error.value)
    sdk[0].get_model.assert_not_called()


def test_replay_only_version_cannot_construct_live_model(inputs, sdk):
    with pytest.raises(InspectJudgeError, match="current Inspect version"):
        validate_collection_environment(
            replace(inputs["options"], inspect_version="0.3.254"),
            {"OPENAI_API_KEY": KEY},
        )
    sdk[0].get_model.assert_not_called()


def test_preflight_rejects_non_openai_grader_and_non_boolean_override(inputs, sdk):
    with pytest.raises(InspectJudgeError, match="OpenAI grader"):
        validate_collection_environment(
            replace(inputs["options"], grader="example-judge"), {}, False
        )
    with pytest.raises(InspectJudgeError, match="must be a boolean"):
        validate_collection_environment(inputs["options"], {}, "false")
    sdk[0].get_model.assert_not_called()


def test_retry_plan_cannot_construct_live_model(inputs, sdk):
    inputs["plan"]["schedule"]["max_attempts"] = 2
    with pytest.raises(InspectJudgeError, match="requires max_attempts=1"):
        asyncio.run(collect_configured(**inputs, environment={"OPENAI_API_KEY": KEY}))
    sdk[0].get_model.assert_not_called()


def test_explicit_construction_passes_strict_checks_and_closes(
    inputs, sdk, monkeypatch
):
    module, model, client = sdk
    result = {"retained": "measurement"}
    stops = []

    async def collect(**kwargs):
        assert kwargs == {**inputs, "model": model, "on_stop": stops.append}
        kwargs["on_stop"]("complete")
        _require_clean_model_configuration(model)
        _require_provider_retries_disabled(model)
        return result

    monkeypatch.setattr(configured, "collect", collect)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-inherited-key")
    assert (
        asyncio.run(
            collect_configured(
                **inputs, environment={"OPENAI_API_KEY": KEY}, on_stop=stops.append
            )
        )
        is result
    )
    module.get_model.assert_called_once_with(
        inputs["options"].grader,
        config=module.GenerateConfig.return_value,
        base_url="https://api.openai.com/v1",
        api_key=KEY,
        responses_api=False,
        max_retries=0,
        memoize=False,
    )
    client.close.assert_awaited_once()
    assert stops == ["complete"]
    assert os.environ["OPENAI_API_KEY"] == "wrong-inherited-key"


@pytest.mark.parametrize(
    "failure", [InspectJudgeError("invalid event"), asyncio.CancelledError()]
)
def test_client_closes_after_collection_failure_and_cancellation(
    inputs, sdk, monkeypatch, failure
):
    monkeypatch.setattr(configured, "collect", AsyncMock(side_effect=failure))
    with pytest.raises(type(failure)):
        asyncio.run(collect_configured(**inputs, environment={"OPENAI_API_KEY": KEY}))
    sdk[2].close.assert_awaited_once()


def test_endpoint_is_verified_before_collection_and_client_is_closed(
    inputs, sdk, monkeypatch
):
    sdk[2].base_url = "https://example.invalid/"
    collect = AsyncMock()
    monkeypatch.setattr(configured, "collect", collect)
    with pytest.raises(InspectJudgeError, match="official OpenAI endpoint"):
        asyncio.run(collect_configured(**inputs, environment={"OPENAI_API_KEY": KEY}))
    collect.assert_not_called()
    sdk[2].close.assert_awaited_once()


def test_model_initialization_error_does_not_expose_key(inputs, sdk):
    sdk[0].get_model.side_effect = RuntimeError(KEY)
    with pytest.raises(InspectJudgeError, match="could not initialize") as error:
        asyncio.run(collect_configured(**inputs, environment={"OPENAI_API_KEY": KEY}))
    assert KEY not in str(error.value)


@pytest.mark.parametrize("collection_fails", [False, True])
def test_close_failure_is_sanitized_and_preserves_collection_error(
    inputs, sdk, monkeypatch, collection_fails
):
    sdk[2].close.side_effect = RuntimeError(KEY)
    monkeypatch.setattr(
        configured,
        "collect",
        AsyncMock(
            return_value={},
            side_effect=InspectJudgeError("invalid event")
            if collection_fails
            else None,
        ),
    )
    with pytest.raises(
        InspectJudgeError,
        match="invalid event" if collection_fails else "could not close",
    ) as error:
        asyncio.run(collect_configured(**inputs, environment={"OPENAI_API_KEY": KEY}))
    assert KEY not in str(error.value)
    sdk[2].close.assert_awaited_once()


def test_missing_provider_client_fails_before_collection(inputs, sdk, monkeypatch):
    sdk[1].api.client = None
    collect = AsyncMock()
    monkeypatch.setattr(configured, "collect", collect)
    with pytest.raises(InspectJudgeError, match="official OpenAI endpoint"):
        asyncio.run(collect_configured(**inputs, environment={"OPENAI_API_KEY": KEY}))
    collect.assert_not_called()


def test_cleanup_finishes_when_cancelled_during_close(inputs, sdk, monkeypatch):
    monkeypatch.setattr(configured, "collect", AsyncMock(return_value={}))

    async def run():
        closing, release = asyncio.Event(), asyncio.Event()
        closed = []

        async def close():
            closing.set()
            await release.wait()
            closed.append(True)

        sdk[2].close = close
        task = asyncio.create_task(
            collect_configured(**inputs, environment={"OPENAI_API_KEY": KEY})
        )
        await closing.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed == [True]

    asyncio.run(run())


def test_real_sdk_constructs_a_strict_model_without_http(inputs, monkeypatch):
    required = os.environ.get("INVARLOCK_REQUIRE_INSPECT_SDK") == "1"
    for name, version in configured._SDK_VERSIONS.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            if required:
                pytest.fail(f"required {name}=={version} is not installed")
            pytest.skip("optional pinned collection SDK is not installed")
    import httpx

    network = AsyncMock(side_effect=AssertionError("real HTTP is forbidden"))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", network)
    models = []

    async def collect(**kwargs):
        model = kwargs["model"]
        models.append(model)
        assert str(model) == inputs["options"].grader
        _require_clean_model_configuration(model)
        _require_provider_retries_disabled(model)
        assert str(model.api.client.base_url) == "https://api.openai.com/v1/"
        return {}

    monkeypatch.setattr(configured, "collect", collect)
    asyncio.run(collect_configured(**inputs, environment={"OPENAI_API_KEY": KEY}))
    assert models[0].api.client.is_closed()
    network.assert_not_called()
