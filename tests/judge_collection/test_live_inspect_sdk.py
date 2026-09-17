"""Exercise the pinned SDK's real request/event conversions without network access."""

from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from invarlock.judge_collection import (
    CollectionOptions,
    RunnerOptions,
    bind_requests,
    collect,
    import_export,
)
from invarlock.judge_measurements.contracts import (
    JudgeMeasurementContractError,
    canonical_payload,
)

FIXTURES = Path(__file__).parent / "fixtures"


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
        assert event["call"]["request"]["tools"] is None
        assert event["call"]["request"]["tool_choice"] is None
        assert event["call"]["response"]["choices"][0]["finish_reason"] == finish_reason
        assert trial["attempts"][0]["finish_reason"] == (
            "max_tokens" if finish_reason == "length" else "stop"
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
    # Check the precise wire projection during offline import too. The native
    # SDK test must not turn missing controls or arbitrary role edits into an
    # allowance for every provider/model.
    mutations = ("role", "temperature", "top_p", "token_limit", "service_tier")
    if service_tier is not None:
        mutations += ("response_tier", "missing_response_tier")
    if reference_mode == "per_case":
        mutations += ("reference",)
    for mutation in mutations:
        changed = copy.deepcopy(samples)
        request = changed[0]["events"][0]["call"]["request"]
        if mutation == "role":
            request["messages"][0]["role"] = (
                "system"
                if judge_model in {"gpt-5.6-sol", "gpt-5.6-luna"}
                else "developer"
            )
        elif mutation == "temperature":
            if judge_model in {"gpt-5.6-sol", "gpt-5.6-luna"}:
                request["temperature"] = 1
            else:
                request.pop("temperature")
        elif mutation == "top_p":
            request.pop("top_p")
        elif mutation == "reference":
            content = json.loads(request["messages"][-1]["content"])
            content["reference"] = "substituted gold"
            request["messages"][-1]["content"] = json.dumps(content)
        elif mutation == "service_tier":
            request["service_tier"] = "priority"
        elif mutation == "response_tier":
            changed[0]["events"][0]["call"]["response"]["service_tier"] = "priority"
        elif mutation == "missing_response_tier":
            changed[0]["events"][0]["call"]["response"].pop("service_tier")
        elif judge_model in {"gpt-5.6-sol", "gpt-5.6-luna"}:
            request["max_tokens"] = request.pop("max_completion_tokens")
        else:
            request["max_tokens"] += 1
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
