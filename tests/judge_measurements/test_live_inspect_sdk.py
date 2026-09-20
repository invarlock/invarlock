"""Exercise the pinned SDK's real request/event conversions without network access."""

from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import json
import os
import socket
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from invarlock.judge_measurements import (
    CollectionOptions,
    RunnerOptions,
    bind_requests,
    collect,
    collect_configured,
    import_export,
)
from invarlock.judge_measurements.contracts import (
    JudgeMeasurementContractError,
    canonical_payload,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _corrupt_wire_response(response, provider, outcome):
    if outcome == "response_model":
        response["modelVersion" if provider == "google" else "model"] = (
            "unapproved-model"
        )
    elif outcome == "response_usage":
        if provider == "google":
            response["usageMetadata"]["promptTokenCount"] = 999
        else:
            response["usage"][
                "input_tokens" if provider == "anthropic" else "prompt_tokens"
            ] = 999
    elif provider == "google":
        response["candidates"][0]["content"]["parts"][0]["text"] = (
            '{"rating":"incorrect"}'
        )
    elif provider == "anthropic":
        response["content"][0]["text"] = '{"rating":"incorrect"}'
    else:
        response["choices"][0]["message"]["content"] = '{"rating":"incorrect"}'


@pytest.mark.parametrize(
    ("provider", "model_name", "credential"),
    [
        ("openai", "gpt-4o-2024-08-06", "OPENAI_API_KEY"),
        ("anthropic", "claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
        ("google", "gemini-2.5-flash", "GOOGLE_API_KEY"),
        ("openrouter", "openai/gpt-4o-mini", "OPENROUTER_API_KEY"),
    ],
)
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "rate_limit",
        "malformed_function",
        "response_model",
        "response_content",
        "response_usage",
    ],
)
def test_configured_provider_wire_collection_replays_without_network(
    tmp_path, monkeypatch, provider, model_name, credential, outcome
):
    from invarlock.judge_measurements import configured

    required = os.environ.get("INVARLOCK_REQUIRE_INSPECT_SDK") == "1"
    for distribution, expected in configured._SDK_VERSIONS.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            if required:
                pytest.fail(f"required {distribution}=={expected} is not installed")
            pytest.skip("optional pinned collection SDK is not installed")
    import aiohttp
    import httpx
    import httpx2
    import inspect_ai.model

    async def forbid_network(*args, **kwargs):
        raise AssertionError("SDK qualification attempted a real network request")

    def forbid_socket(*args, **kwargs):
        pytest.fail("SDK qualification attempted a real socket request")

    # Exercise the real configured scope without granting actual connectivity.
    # These socket/transport blockers do not consult its ContextVar permission.
    for name in ("connect", "connect_ex", "sendto", "sendmsg"):
        if hasattr(socket.socket, name):
            monkeypatch.setattr(socket.socket, name, forbid_socket)
    monkeypatch.setattr(socket, "create_connection", forbid_socket)
    monkeypatch.setattr(socket, "getaddrinfo", forbid_socket)
    monkeypatch.setenv("INVARLOCK_ALLOW_JUDGE_NETWORK", "1")

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", forbid_network
    )
    monkeypatch.setattr(
        httpx2.AsyncHTTPTransport, "handle_async_request", forbid_network
    )
    monkeypatch.setattr(
        httpx.HTTPTransport,
        "handle_request",
        lambda *a, **k: pytest.fail("real HTTP is forbidden"),
    )
    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbid_network)
    for config in configured._PROVIDERS.values():
        for name in (*config.endpoint_variables, *config.forbidden_variables):
            monkeypatch.delenv(name, raising=False)
    documents = {
        name: json.loads((FIXTURES / f"{name}.json").read_text())
        for name in ("plan", "export", "frozen", "baseline_run", "subject_run")
    }
    grader = f"{provider}/{model_name}"
    documents["plan"]["judge"].update(
        provider=provider,
        requested_model=grader,
        approved_resolved_models=[model_name],
    )
    plan = bind_requests(documents["plan"], documents["frozen"])
    options = replace(
        CollectionOptions.from_mapping(documents["export"]["collection"]),
        grader=grader,
        requests_per_minute=10000,
        max_calls=2,
        max_input_tokens=200,
        max_output_tokens=256,
        max_cost_microusd=200,
    )
    requests = []

    def respond(request):
        requests.append(request)
        number = len(requests)
        transport_module = httpx2 if provider == "anthropic" else httpx
        if outcome == "rate_limit":
            return transport_module.Response(
                429,
                json={
                    "error": {
                        "code": 429,
                        "type": "rate_limit_error",
                        "message": "offline quota",
                    }
                },
            )
        if provider == "google":
            response = {
                "responseId": f"google-{number}",
                "modelVersion": model_name,
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": '{"rating":"correct"}'}],
                        },
                        "finishReason": "MALFORMED_FUNCTION_CALL"
                        if outcome == "malformed_function"
                        else "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 30,
                    "cachedContentTokenCount": 10,
                    "candidatesTokenCount": 3,
                    "thoughtsTokenCount": 2,
                    "totalTokenCount": 35,
                },
            }
        elif provider == "anthropic":
            response = {
                "id": f"anthropic-{number}",
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [{"type": "text", "text": '{"rating":"correct"}'}],
                "stop_reason": "pause_turn"
                if outcome == "malformed_function"
                else "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 20,
                    "cache_read_input_tokens": 7,
                    "cache_creation_input_tokens": 3,
                    "output_tokens": 5,
                },
            }
        else:
            response = {
                "id": f"openrouter-{number}",
                "object": "chat.completion",
                "created": 0,
                "model": model_name,
                "service_tier": "default",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"rating":"correct"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 5,
                    "total_tokens": 35,
                    "prompt_tokens_details": {"cached_tokens": 10},
                },
            }
        return transport_module.Response(200, json=response)

    original_get_model = inspect_ai.model.get_model
    original_clients = []
    from invarlock.judge_measurements import runner as live_runner

    original_project = live_runner._project_event

    def project(event, **kwargs):
        if outcome == "success":
            assert event.error is None
        if outcome.startswith("response_"):
            assert event.error is None
            _corrupt_wire_response(event.call.response, provider, outcome)
        return original_project(event, **kwargs)

    monkeypatch.setattr(live_runner, "_project_event", project)

    def get_model(*args, **kwargs):
        model = original_get_model(*args, **kwargs)
        if provider == "google":
            original_client = model.api.model_client

            def model_client(http_options):
                assert http_options.retry_options.attempts == 1
                http_options.httpx_async_client = httpx.AsyncClient(
                    transport=httpx.MockTransport(respond)
                )
                return original_client(http_options)

            model.api.model_client = model_client
        else:
            sdk_module = importlib.import_module(
                "anthropic" if provider == "anthropic" else "openai"
            )
            client_type = (
                sdk_module.AsyncAnthropic
                if provider == "anthropic"
                else sdk_module.AsyncOpenAI
            )
            transport_module = httpx2 if provider == "anthropic" else httpx
            original_clients.append(model.api.client)
            model.api.client = client_type(
                api_key="offline-test-key",
                max_retries=0,
                base_url=configured._PROVIDERS[provider].base_url,
                http_client=transport_module.AsyncClient(
                    transport=transport_module.MockTransport(respond)
                ),
            )
        return model

    monkeypatch.setattr(inspect_ai.model, "get_model", get_model)
    runner = RunnerOptions(tmp_path / "checkpoint", "correctness", 30)

    async def run():
        args = {
            "plan": plan,
            "options": options,
            "runner": runner,
            "baseline_run": documents["baseline_run"],
            "subject_run": documents["subject_run"],
            "environment": {credential: "offline-test-key"},
        }
        try:
            measurements = await collect_configured(**args)
            assert await collect_configured(**args) == measurements
            return measurements
        finally:
            for client in original_clients:
                await client.close()

    if outcome.startswith("response_"):
        from invarlock.judge_measurements import InspectJudgeError

        with pytest.raises(InspectJudgeError, match="contradicts"):
            asyncio.run(run())
        assert 1 <= len(requests) <= 2
        assert not list(runner.checkpoint_directory.glob("result-*.json"))
        return
    measurements = asyncio.run(run())
    assert len(requests) == 2
    assert b"offline-test-key" not in canonical_payload(measurements)
    for request in requests:
        assert str(request.url).startswith(
            configured._PROVIDERS[provider].base_url + "/"
        )
        wire = json.loads(request.content)
        assert wire.get("tools") in (None, [])
        if provider == "google":
            assert wire["generationConfig"]["maxOutputTokens"] == 128
        else:
            assert wire["model"] == model_name
            if provider == "anthropic":
                assert "top_p" not in wire
                assert wire["temperature"] == 0
    successful = outcome != "rate_limit" and not (
        provider in {"google", "anthropic"} and outcome == "malformed_function"
    )
    for trial in measurements["trials"]:
        assert len(trial["attempts"]) == 1
        assert trial["parse"]["status"] == ("ok" if successful else "unavailable")
        if successful:
            assert trial["attempts"][0]["usage"] == {
                "input_tokens": 30,
                "output_tokens": 5,
            }
            assert trial["attempts"][0]["request_id"]


@pytest.mark.parametrize("reference_mode", [None, "per_case"])
@pytest.mark.parametrize("service_tier", [None, "default"])
@pytest.mark.parametrize(
    "judge_model", ["gpt-4o-2024-08-06", "gpt-5.6-sol", "gpt-5.6-luna"]
)
@pytest.mark.parametrize(
    ("finish_reason", "completion", "parse_status"),
    [
        ("stop", '{"rating":"correct"}', "ok"),
        ("length", '{"rating":"correct"}', "ok"),
        ("length", '{"rating":', "invalid"),
    ],
)
def test_live_inspect_chat_completion_replays_offline(
    tmp_path,
    monkeypatch,
    finish_reason,
    completion,
    parse_status,
    judge_model,
    reference_mode,
    service_tier,
):
    required = os.environ.get("INVARLOCK_REQUIRE_INSPECT_SDK") == "1"
    try:
        version = importlib.metadata.version("inspect-ai")
    except importlib.metadata.PackageNotFoundError:
        if required:
            pytest.fail("required pinned Inspect SDK is not installed")
        pytest.skip("optional pinned Inspect SDK is not installed")
    if version != "0.3.263":
        if required:
            pytest.fail("required Inspect SDK version differs from qualification")
        pytest.skip("this conversion test requires Inspect 0.3.263")
    loader = importlib.import_module if required else pytest.importorskip
    model_module = loader("inspect_ai.model")
    openai = loader("openai")
    httpx = loader("httpx")
    assert importlib.metadata.version("openai") == "3.13.0"
    assert importlib.metadata.version("httpx") == "0.28.1"

    async def forbid_network(*args, **kwargs):
        raise AssertionError("live SDK test attempted a real HTTP request")

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", forbid_network
    )
    plan, exported, frozen = [
        json.loads((FIXTURES / f"{name}.json").read_text())
        for name in ("plan", "export", "frozen")
    ]
    runs = {
        f"{side}_run": json.loads((FIXTURES / f"{side}_run.json").read_text())
        for side in ("baseline", "subject")
    }
    if reference_mode is not None:
        plan["prompt"]["reference_mode"] = reference_mode
        for row in runs["baseline_run"]["records"]:
            frozen[row["id"]]["expected"] = row["expected"]
    plan["judge"].update(
        provider="openai",
        requested_model=f"openai/{judge_model}",
        approved_resolved_models=[judge_model],
    )
    if judge_model in {"gpt-5.6-sol", "gpt-5.6-luna"}:
        plan["judge"]["config"].update(
            temperature="1",
            reasoning_effort="none" if judge_model == "gpt-5.6-sol" else "xhigh",
            **({"max_output_tokens": 25000} if judge_model == "gpt-5.6-luna" else {}),
        )
    plan = bind_requests(plan, frozen)
    options = replace(
        CollectionOptions.from_mapping(exported["collection"]),
        grader=plan["judge"]["requested_model"],
        requests_per_minute=10000,
        **({"max_output_tokens": 50000} if judge_model == "gpt-5.6-luna" else {}),
    )
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": judge_model,
                "service_tier": "default",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": completion},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 5,
                    "total_tokens": 35,
                },
            },
        )

    async def run():
        model = model_module.get_model(
            options.grader,
            api_key="fake-key-for-offline-test",
            responses_api=False,
            **({"service_tier": service_tier} if service_tier is not None else {}),
            max_retries=0,
            memoize=False,
        )
        original_client = model.api.client
        model.api.client = openai.AsyncOpenAI(
            api_key="fake-key-for-offline-test",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        runner = RunnerOptions(tmp_path / "checkpoint", "correctness", 30)
        try:
            measurements = await collect(
                plan=plan, options=options, runner=runner, model=model, **runs
            )
            assert len(requests) == 2
            resumed = await collect(
                plan=plan, options=options, runner=runner, model=model, **runs
            )
            assert resumed == measurements
            assert len(requests) == 2
            return measurements
        finally:
            await model.api.client.close()
            await original_client.close()

    measurements = asyncio.run(run())
    for request in requests:
        if service_tier is None:
            assert "service_tier" not in request
        else:
            assert request["service_tier"] == "default"
        content = json.loads(request["messages"][-1]["content"])
        assert content["input"] == runs["baseline_run"]["records"][0]["input"]
        if reference_mode == "per_case":
            assert (
                content["reference"] == runs["baseline_run"]["records"][0]["expected"]
            )
        else:
            assert "reference" not in content
        if judge_model in {"gpt-5.6-sol", "gpt-5.6-luna"}:
            assert request["messages"][0]["role"] == "developer"
            assert "temperature" not in request
            assert request["max_completion_tokens"] == (
                25000 if judge_model == "gpt-5.6-luna" else 128
            )
            assert "max_tokens" not in request
            assert request["reasoning_effort"] == (
                "xhigh" if judge_model == "gpt-5.6-luna" else "none"
            )
        else:
            assert request["messages"][0]["role"] == "system"
            assert request["temperature"] == 0
            assert request["max_tokens"] == 128
    source = json.loads(measurements["sources"][0]["content"])
    samples = []
    for record in source["records"]:
        trial = record["trial"]
        event = record["events"][0]
        assert event["call"]["request"]["format"] == "invarlock/judge-request-v1"
        assert event["call"]["request"]["tools"] == []
        assert event["call"]["response"]["format"] == (
            "invarlock/judge-provider-response-v1"
        )
        assert trial["attempts"][0]["finish_reason"] == (
            "max_tokens" if finish_reason == "length" else "stop"
        )
        assert (
            event["call"]["response"]["finish_reason"]
            == trial["attempts"][0]["finish_reason"]
        )
        assert trial["parse"]["status"] == parse_status
        samples.append(
            {
                "id": trial["trial_id"],
                "epoch": 1,
                "metadata": {
                    key: trial[key]
                    for key in (
                        "case_id",
                        "side",
                        "repetition",
                        "plan_sha256",
                        "answer_sha256",
                    )
                }
                | {"scorer_id": "correctness"},
                "events": record["events"],
            }
        )
    imported = import_export(
        canonical_payload(
            {
                "format": exported["format"],
                "profile": exported["profile"],
                "inspect_version": version,
                "collection": asdict(options),
                "samples": samples,
            }
        ),
        plan=plan,
        options=options,
        **runs,
    )
    assert imported == measurements
    # Check the provider-neutral retained projection during offline import too.
    mutations = (
        "role",
        "temperature",
        "top_p",
        "token_limit",
        "response_model",
        "response_id",
        "response_finish",
        "response_usage",
    )
    if reference_mode == "per_case":
        mutations += ("reference",)
    for mutation in mutations:
        changed = copy.deepcopy(samples)
        request = changed[0]["events"][0]["call"]["request"]
        if mutation == "role":
            request["messages"][0]["role"] = "developer"
        elif mutation == "temperature":
            request["config"]["temperature"] = "0.5"
        elif mutation == "top_p":
            request["config"]["top_p"] = "0.5"
        elif mutation == "reference":
            content = json.loads(request["messages"][-1]["content"])
            content["reference"] = "substituted gold"
            request["messages"][-1]["content"] = json.dumps(content)
        elif mutation == "response_model":
            changed[0]["events"][0]["call"]["response"]["model"] = "changed"
        elif mutation == "response_id":
            changed[0]["events"][0]["call"]["response"]["id"] = "changed"
        elif mutation == "response_finish":
            changed[0]["events"][0]["call"]["response"]["finish_reason"] = "changed"
        elif mutation == "response_usage":
            changed[0]["events"][0]["call"]["response"]["usage"]["input_tokens"] += 1
        else:
            request["config"]["max_output_tokens"] += 1
        with pytest.raises(JudgeMeasurementContractError, match="provider|projection"):
            import_export(
                canonical_payload(
                    {
                        "format": exported["format"],
                        "profile": exported["profile"],
                        "inspect_version": version,
                        "collection": asdict(options),
                        "samples": changed,
                    }
                ),
                plan=plan,
                options=options,
                **runs,
            )
