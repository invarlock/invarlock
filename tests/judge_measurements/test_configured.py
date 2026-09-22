"""Configured collection fails before calls and always releases its client."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import socket
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from invarlock import security
from invarlock.judge_measurements import (
    CollectionOptions,
    InspectJudgeError,
    RunnerOptions,
    bind_requests,
    collect_configured,
    configured,
    validate_collection_environment,
)
from invarlock.judge_measurements.runner import (
    _require_clean_model_configuration,
    _require_provider_retries_disabled,
)

FIXTURES = Path(__file__).with_name("fixtures")
KEY = "offline-test-key"


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    # These SDK/model doubles exercise collection after admission policy passes;
    # do not remove the real process socket guard for offline tests.
    monkeypatch.setattr(configured, "network_policy_allows", lambda: True)
    monkeypatch.delenv("INVARLOCK_ALLOW_JUDGE_NETWORK", raising=False)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_SAFETY_IDENTIFIER",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_BASE_URL",
        "GOOGLE_VERTEX_BASE_URL",
        "VERTEX_BASE_URL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_USE_ADC",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "VERTEX_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GENAI_CLIENT_MODE",
        "GOOGLE_GENAI_REPLAYS_DIRECTORY",
        "GOOGLE_GENAI_REPLAY_ID",
    ):
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
        messages=SimpleNamespace(create=AsyncMock()),
    )

    async def close_api():
        await client.close()

    model = SimpleNamespace(
        config=config,
        model_args={
            "max_retries": 0,
            "responses_api": False,
            "service_tier": "default",
        },
        api=SimpleNamespace(
            client=client,
            responses_api=False,
            service_tier="default",
            model_args={},
            aclose=close_api,
        ),
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


@pytest.mark.parametrize(
    "environment_flag",
    [None, "INVARLOCK_ALLOW_NETWORK", "INVARLOCK_ALLOW_JUDGE_NETWORK"],
)
def test_denied_network_stops_before_sdk_loading_or_admission(
    inputs, sdk, monkeypatch, environment_flag
):
    module, _, client = sdk
    monkeypatch.setattr(
        configured, "network_policy_allows", security.network_policy_allows
    )
    validate = Mock(side_effect=AssertionError("SDK validation reached"))
    collect = AsyncMock(side_effect=AssertionError("call admission reached"))
    monkeypatch.setattr(configured, "validate_collection_environment", validate)
    monkeypatch.setattr(configured, "collect", collect)
    stopped = Mock()
    environment = {"OPENAI_API_KEY": KEY}
    if environment_flag is not None:
        # Credential mappings cannot override an already enforced process policy.
        environment[environment_flag] = "1"
    previous = security.network_policy_allows()
    security.enforce_network_policy(False)
    try:
        with pytest.raises(InspectJudgeError, match="INVARLOCK_ALLOW_JUDGE_NETWORK=1"):
            asyncio.run(
                collect_configured(**inputs, environment=environment, on_stop=stopped)
            )
        assert not security.network_policy_allows()
    finally:
        security.enforce_network_policy(previous)
    validate.assert_not_called()
    configured.importlib.import_module.assert_not_called()
    module.get_model.assert_not_called()
    client.close.assert_not_called()
    collect.assert_not_called()
    stopped.assert_not_called()
    assert not inputs["runner"].checkpoint_directory.exists()


def test_offline_preflight_and_explicit_collection_scope_preserve_policy(
    inputs, sdk, monkeypatch
):
    module, _, client = sdk
    monkeypatch.setattr(
        configured, "network_policy_allows", security.network_policy_allows
    )
    collected = AsyncMock(return_value={"offline_double": True})
    monkeypatch.setattr(configured, "collect", collected)
    previous = security.network_policy_allows()
    security.enforce_network_policy(False)
    try:
        metadata = validate_collection_environment(
            inputs["options"], {"OPENAI_API_KEY": KEY}
        )
        assert metadata["credential_available"] is True
        module.get_model.assert_not_called()
        assert not security.network_policy_allows()
        with security.temporarily_allow_network():
            assert asyncio.run(
                collect_configured(**inputs, environment={"OPENAI_API_KEY": KEY})
            ) == {"offline_double": True}
        assert not security.network_policy_allows()
    finally:
        security.enforce_network_policy(previous)
    collected.assert_awaited_once()
    client.close.assert_awaited_once()
    assert not inputs["runner"].checkpoint_directory.exists()


@pytest.mark.parametrize(
    "outcome",
    ["complete", "initialize_error", "collect_error", "close_error", "cancelled"],
)
def test_judge_opt_in_is_task_local_and_restored_after_every_lifecycle_exit(
    inputs, sdk, monkeypatch, outcome
):
    from invarlock.evaluation_transaction import _require_closed_runtime_switches

    module, model, client = sdk
    for name in (
        "INVARLOCK_ALLOW_NETWORK",
        "INVARLOCK_ALLOW_REMOTE_CODE",
        "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INVARLOCK_ALLOW_JUDGE_NETWORK", "1")
    monkeypatch.setattr(
        configured, "network_policy_allows", security.network_policy_allows
    )

    def get_model(*args, **kwargs):
        assert security.network_policy_allows()
        if outcome == "initialize_error":
            raise ValueError("offline initialization failure")
        return model

    async def close():
        assert security.network_policy_allows()
        if outcome == "close_error":
            raise ValueError("offline cleanup failure")

    module.get_model.side_effect = get_model
    client.close.side_effect = close

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def collect(**kwargs):
            assert security.network_policy_allows()
            entered.set()
            await release.wait()
            if outcome == "collect_error":
                raise ValueError("offline collection failure")
            return {"offline_double": True}

        async def invoke():
            try:
                return await collect_configured(
                    **inputs, environment={"OPENAI_API_KEY": KEY}
                )
            finally:
                # The same calling task must regain its original denied policy.
                assert not security.network_policy_allows()

        monkeypatch.setattr(configured, "collect", collect)
        task = asyncio.create_task(invoke())
        if outcome != "initialize_error":
            await asyncio.wait_for(entered.wait(), timeout=2)
            # This concurrent task and native runtime admission stay offline.
            assert not security.network_policy_allows()
            _require_closed_runtime_switches()
            with socket.socket() as blocked_socket:
                with pytest.raises(RuntimeError, match="Network access disabled"):
                    blocked_socket.connect(("127.0.0.1", 1))
            if outcome == "cancelled":
                task.cancel()
            else:
                release.set()
        if outcome == "complete":
            assert await task == {"offline_double": True}
        else:
            expected = (
                asyncio.CancelledError
                if outcome == "cancelled"
                else (ValueError if outcome == "collect_error" else InspectJudgeError)
            )
            with pytest.raises(expected):
                await task
        assert not security.network_policy_allows()

    previous = security.network_policy_allows()
    security.enforce_default_security()
    try:
        # The scoped opt-in does not relax native capture's runtime switches.
        _require_closed_runtime_switches()
        assert not security.network_policy_allows()
        validate_collection_environment(inputs["options"], {"OPENAI_API_KEY": KEY})
        module.get_model.assert_not_called()
        assert not security.network_policy_allows()
        asyncio.run(scenario())
        assert not security.network_policy_allows()
    finally:
        security.enforce_network_policy(previous)
    assert client.close.await_count == (0 if outcome == "initialize_error" else 1)
    assert not inputs["runner"].checkpoint_directory.exists()


def test_preflight_metadata_is_usable_and_never_constructs_model(inputs, sdk):
    module, _, _ = sdk
    metadata = validate_collection_environment(
        inputs["options"], {"OPENAI_API_KEY": KEY}
    )
    assert metadata == {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "service_tier": "default",
        "credential_variable": "OPENAI_API_KEY",
        "sdk_versions": {
            "inspect-ai": "0.3.263",
            "openai": "3.13.0",
            "httpx": "0.28.1",
            "httpx2": "2.12.0",
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


@pytest.mark.parametrize(
    ("provider", "credential", "distribution", "version", "base_url"),
    [
        (
            "openai",
            "OPENAI_API_KEY",
            "openai",
            "3.13.0",
            "https://api.openai.com/v1",
        ),
        (
            "anthropic",
            "ANTHROPIC_API_KEY",
            "anthropic",
            "1.6.0",
            "https://api.anthropic.com",
        ),
        (
            "google",
            "GOOGLE_API_KEY",
            "google-genai",
            "2.24.0",
            "https://generativelanguage.googleapis.com",
        ),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "openai",
            "3.13.0",
            "https://openrouter.ai/api/v1",
        ),
    ],
)
def test_preflight_selects_credentials_and_sdk_by_grader_provider(
    inputs, sdk, provider, credential, distribution, version, base_url
):
    options = replace(inputs["options"], grader=f"{provider}/approved-model")
    metadata = validate_collection_environment(options, {credential: KEY})
    assert metadata["provider"] == provider
    assert metadata["credential_variable"] == credential
    assert metadata["base_url"] == base_url
    assert metadata["sdk_versions"] == {
        "inspect-ai": "0.3.263",
        "httpx": "0.28.1",
        "httpx2": "2.12.0",
        distribution: version,
    }
    assert metadata["credential_available"] is True
    assert KEY not in json.dumps(metadata)
    sdk[0].get_model.assert_not_called()


def test_google_accepts_gemini_key_alias(inputs, sdk):
    options = replace(inputs["options"], grader="google/approved-model")
    metadata = validate_collection_environment(options, {"GEMINI_API_KEY": KEY})
    assert metadata["credential_variable"] == "GEMINI_API_KEY"
    assert metadata["credential_available"] is True


@pytest.mark.parametrize(
    "grader",
    [
        "google/vertex/gemini-2.5-flash",
        "anthropic/bedrock/model",
        "openai/azure/model",
        "google/",
    ],
)
def test_configured_collection_rejects_alternate_service_model_paths(
    inputs, sdk, grader
):
    with pytest.raises(InspectJudgeError, match="direct API|supported provider"):
        validate_collection_environment(
            replace(inputs["options"], grader=grader), {}, False
        )
    sdk[0].get_model.assert_not_called()


@pytest.mark.parametrize("provider", ["anthropic", "google"])
def test_configured_collection_rejects_ignored_seeds_before_construction(
    inputs, sdk, provider
):
    grader = f"{provider}/model"
    inputs["plan"]["judge"].update(
        provider=provider, requested_model=grader, approved_resolved_models=["model"]
    )
    inputs["plan"]["judge"]["config"]["seed"] = 42
    inputs["options"] = replace(inputs["options"], grader=grader)
    with pytest.raises(InspectJudgeError, match="does not support seed"):
        asyncio.run(collect_configured(**inputs, environment={}))
    sdk[0].get_model.assert_not_called()


def test_anthropic_profile_rejects_combined_sampling_controls(inputs, sdk):
    grader = "anthropic/claude-sonnet-4-5"
    inputs["plan"]["judge"].update(
        provider="anthropic",
        requested_model=grader,
        approved_resolved_models=["claude-sonnet-4-5"],
    )
    inputs["plan"]["judge"]["config"]["top_p"] = "0.8"
    inputs["options"] = replace(inputs["options"], grader=grader)
    with pytest.raises(InspectJudgeError, match="requires top_p=1"):
        asyncio.run(collect_configured(**inputs, environment={}))
    sdk[0].get_model.assert_not_called()


@pytest.mark.parametrize(
    ("provider", "model", "limit", "message"),
    [
        ("anthropic", "claude-sonnet-4-5", 128, "reasoning budget"),
        ("google", "gemini-2.5-flash", 128, "reasoning budget"),
        (
            "anthropic",
            "claude-sonnet-4-6",
            20000,
            "not qualified for this model",
        ),
        ("google", "gemini-3-flash", 20000, "not qualified for this model"),
    ],
)
def test_provider_reasoning_preflight_rejects_unbound_or_impossible_profiles(
    inputs, sdk, provider, model, limit, message
):
    grader = f"{provider}/{model}"
    inputs["plan"]["judge"].update(
        provider=provider,
        requested_model=grader,
        approved_resolved_models=[model],
    )
    inputs["plan"]["judge"]["config"].update(
        reasoning_effort="high", max_output_tokens=limit, temperature="1"
    )
    inputs["options"] = replace(
        inputs["options"], grader=grader, max_output_tokens=40000
    )
    with pytest.raises(InspectJudgeError, match=message):
        asyncio.run(collect_configured(**inputs, environment={}))
    sdk[0].get_model.assert_not_called()


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "claude-sonnet-4-7"),
        ("google", "gemini-3-flash"),
        ("anthropic", "custom-alias"),
        ("google", "custom-alias"),
    ],
)
@pytest.mark.parametrize("reasoning_effort", [None, "none"])
def test_unqualified_provider_family_stops_before_model_construction(
    inputs, sdk, provider, model, reasoning_effort
):
    grader = f"{provider}/{model}"
    inputs["plan"]["judge"].update(
        provider=provider,
        requested_model=grader,
        approved_resolved_models=[model],
    )
    inputs["plan"]["judge"]["config"].update(
        reasoning_effort=reasoning_effort, temperature="1"
    )
    inputs["options"] = replace(inputs["options"], grader=grader)
    with pytest.raises(InspectJudgeError, match="live wire shape is not qualified"):
        asyncio.run(collect_configured(**inputs, environment={}))
    sdk[0].get_model.assert_not_called()


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "claude-sonnet-4-7"),
        ("google", "gemini-3-flash"),
        ("anthropic", "custom-alias"),
        ("google", "custom-alias"),
    ],
)
@pytest.mark.parametrize("reasoning_effort", [None, "none"])
def test_public_preflights_reject_unqualified_provider_before_collection(
    inputs, provider, model, reasoning_effort
):
    from invarlock.judge_measurements import native_workflow, workflow

    grader = f"{provider}/{model}"
    inputs["plan"]["judge"].update(
        provider=provider,
        requested_model=grader,
        approved_resolved_models=[model],
    )
    inputs["plan"]["judge"]["config"].update(
        reasoning_effort=reasoning_effort, temperature="1"
    )
    configuration = asdict(replace(inputs["options"], grader=grader))
    with pytest.raises(workflow.JudgeWorkflowError, match="not qualified"):
        workflow._collection_budgets(configuration, inputs["plan"])
    with pytest.raises(native_workflow.JudgeWorkflowError, match="not qualified"):
        native_workflow.collection_preflight(configuration)


def test_unbounded_google_client_is_rejected_before_calls():
    class Model:
        api = SimpleNamespace()

        def __str__(self):
            return "google/gemini-2.5-flash"

    with pytest.raises(InspectJudgeError, match="single-attempt client"):
        _require_provider_retries_disabled(Model())


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "openrouter"])
def test_selected_provider_rejects_all_ambient_controls(
    inputs, sdk, monkeypatch, provider
):
    options = replace(inputs["options"], grader=f"{provider}/model")
    config = configured._PROVIDERS[provider]
    for name in (*config.endpoint_variables, *config.forbidden_variables):
        with monkeypatch.context() as patch:
            patch.setenv(name, "unapproved")
            with pytest.raises(InspectJudgeError, match="unsupported"):
                validate_collection_environment(options, {}, False)
    sdk[0].get_model.assert_not_called()


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


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("value", ["", "private-identifier"])
def test_safety_identifier_override_fails_before_calls(
    inputs, sdk, monkeypatch, inherited, value
):
    environment = {"OPENAI_API_KEY": KEY}
    if inherited:
        monkeypatch.setenv("OPENAI_SAFETY_IDENTIFIER", value)
    else:
        environment["OPENAI_SAFETY_IDENTIFIER"] = value
    with pytest.raises(InspectJudgeError, match="request controls") as error:
        asyncio.run(collect_configured(**inputs, environment=environment))
    assert "private-identifier" not in str(error.value)
    sdk[0].get_model.assert_not_called()
    assert not inputs["runner"].checkpoint_directory.exists()


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


def test_preflight_rejects_unsupported_grader_and_non_boolean_override(inputs, sdk):
    with pytest.raises(InspectJudgeError, match="supported provider"):
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
        service_tier="default",
        max_retries=0,
        memoize=False,
    )
    client.close.assert_awaited_once()
    assert stops == ["complete"]
    assert os.environ["OPENAI_API_KEY"] == "wrong-inherited-key"


@pytest.mark.parametrize(
    ("provider", "credential", "base_url", "extra_model_args"),
    [
        (
            "anthropic",
            "ANTHROPIC_API_KEY",
            "https://api.anthropic.com",
            {"max_retries": 0},
        ),
        ("google", "GOOGLE_API_KEY", "https://generativelanguage.googleapis.com", {}),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1",
            {"responses_api": False, "max_retries": 0},
        ),
    ],
)
def test_configured_collection_constructs_selected_provider(
    inputs,
    sdk,
    monkeypatch,
    provider,
    credential,
    base_url,
    extra_model_args,
):
    module, model, client = sdk
    model_name = {
        "anthropic": "claude-sonnet-4-5",
        "google": "gemini-2.5-flash",
    }.get(provider, "approved-model")
    grader = f"{provider}/{model_name}"
    inputs["plan"]["judge"].update(
        provider=provider,
        requested_model=grader,
        approved_resolved_models=[model_name],
    )
    inputs["options"] = replace(inputs["options"], grader=grader)
    model.api.service_tier = None
    client.base_url = base_url
    if provider == "google":
        model.api.client = None
        model.api.base_url = base_url
        model.api.model_client = Mock()
    elif provider == "anthropic":
        model.api.generate = AsyncMock()
        model.api.is_claude_4_7_or_later = Mock(return_value=False)
        model.api.is_using_thinking = Mock(return_value=False)
    result = {"retained": provider}
    monkeypatch.setattr(configured, "collect", AsyncMock(return_value=result))

    assert (
        asyncio.run(collect_configured(**inputs, environment={credential: KEY}))
        == result
    )
    expected = {
        "config": module.GenerateConfig.return_value,
        "api_key": KEY,
        "memoize": False,
        **extra_model_args,
    }
    if base_url is not None:
        expected["base_url"] = base_url
    module.get_model.assert_called_once_with(grader, **expected)
    client.close.assert_awaited_once()


@pytest.mark.parametrize("tier", [None, "auto", "priority", "flex", "ultrafast"])
def test_configured_tier_is_verified_before_collection(inputs, sdk, monkeypatch, tier):
    sdk[1].api.service_tier = tier
    collect = AsyncMock()
    monkeypatch.setattr(configured, "collect", collect)
    with pytest.raises(InspectJudgeError, match="standard service tier"):
        asyncio.run(collect_configured(**inputs, environment={"OPENAI_API_KEY": KEY}))
    collect.assert_not_called()
    sdk[2].close.assert_awaited_once()
    assert not inputs["runner"].checkpoint_directory.exists()


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
    with pytest.raises(InspectJudgeError, match="official openai endpoint"):
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
    with pytest.raises(InspectJudgeError, match="official openai endpoint"):
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


@pytest.mark.parametrize(
    ("provider", "grader", "credential"),
    [
        ("anthropic", "anthropic/claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
        ("google", "google/gemini-2.5-flash", "GOOGLE_API_KEY"),
        ("openrouter", "openrouter/openai/gpt-4o-mini", "OPENROUTER_API_KEY"),
    ],
)
def test_real_non_openai_sdks_construct_without_http(
    inputs, monkeypatch, provider, grader, credential
):
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
    inputs["plan"]["judge"].update(
        provider=provider,
        requested_model=grader,
        approved_resolved_models=[grader.split("/", 1)[1]],
    )
    inputs["options"] = replace(inputs["options"], grader=grader)
    models = []

    async def collect(**kwargs):
        model = kwargs["model"]
        models.append(model)
        assert str(model) == grader
        _require_clean_model_configuration(model)
        _require_provider_retries_disabled(model)
        return {}

    monkeypatch.setattr(configured, "collect", collect)
    asyncio.run(collect_configured(**inputs, environment={credential: KEY}))
    assert len(models) == 1
    client = getattr(models[0].api, "client", None)
    if client is not None:
        assert client.is_closed()
    network.assert_not_called()


def test_provider_without_async_cleanup_is_rejected():
    with pytest.raises(InspectJudgeError, match="async cleanup"):
        asyncio.run(configured._close_model(SimpleNamespace(api=SimpleNamespace())))


def test_cancelled_provider_cleanup_preserves_cancellation():
    async def close():
        raise asyncio.CancelledError

    model = SimpleNamespace(api=SimpleNamespace(aclose=close))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(configured._close_model(model))


def test_anthropic_unqualified_model_cannot_dispatch(inputs, sdk, monkeypatch):
    grader = "anthropic/claude-sonnet-4-7"
    inputs["plan"]["judge"].update(
        provider="anthropic",
        requested_model=grader,
        approved_resolved_models=["claude-sonnet-4-7"],
    )
    inputs["options"] = replace(inputs["options"], grader=grader)
    sdk[1].api.generate = AsyncMock()
    sdk[1].api.is_claude_4_7_or_later = lambda: True
    collect = AsyncMock()
    monkeypatch.setattr(configured, "collect", collect)
    with pytest.raises(InspectJudgeError, match="live wire shape is not qualified"):
        asyncio.run(
            collect_configured(**inputs, environment={"ANTHROPIC_API_KEY": KEY})
        )
    sdk[0].get_model.assert_not_called()
    collect.assert_not_called()
    sdk[1].api.generate.assert_not_called()
    sdk[2].messages.create.assert_not_called()


def test_configured_graceful_batch_stop_preserves_runner_and_closes_client(
    inputs, sdk, monkeypatch
):
    _, _, client = sdk
    inputs["runner"] = replace(inputs["runner"], stop_after_batches=1)
    retained = {"retained": "completed-batch"}
    stops = []

    async def collect_batch(**arguments):
        assert arguments["runner"] is inputs["runner"]
        assert arguments["runner"].stop_after_batches == 1
        arguments["on_stop"]("requested")
        return retained

    monkeypatch.setattr(configured, "collect", collect_batch)
    assert (
        asyncio.run(
            collect_configured(
                **inputs, environment={"OPENAI_API_KEY": KEY}, on_stop=stops.append
            )
        )
        == retained
    )
    assert stops == ["requested"]
    client.close.assert_awaited_once()
