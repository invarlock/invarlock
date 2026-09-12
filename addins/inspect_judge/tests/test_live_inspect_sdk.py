"""Exercise the pinned SDK's real request/event conversions without network access."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from invarlock_addins.inspect_judge import (
    CollectionOptions,
    RunnerOptions,
    bind_requests,
    collect,
    import_export,
)

from invarlock.judge_measurements.contracts import canonical_payload

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("finish_reason", "completion", "parse_status"),
    [
        ("stop", '{"rating":"correct"}', "ok"),
        ("length", '{"rating":"correct"}', "ok"),
        ("length", '{"rating":', "invalid"),
    ],
)
def test_live_inspect_chat_completion_replays_offline(
    tmp_path, monkeypatch, finish_reason, completion, parse_status
):
    try:
        version = importlib.metadata.version("inspect-ai")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("optional pinned Inspect SDK is not installed")
    if version != "0.3.254":
        pytest.skip("this conversion test requires Inspect 0.3.254")
    model_module = pytest.importorskip("inspect_ai.model")
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")

    async def forbid_network(*args, **kwargs):
        raise AssertionError("live SDK test attempted a real HTTP request")

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", forbid_network
    )
    plan, exported, frozen = [
        json.loads((FIXTURES / f"{name}.json").read_text())
        for name in ("plan", "export", "frozen")
    ]
    plan["judge"].update(
        provider="openai",
        requested_model="openai/gpt-4o-2024-08-06",
        approved_resolved_models=["gpt-4o-2024-08-06"],
    )
    plan = bind_requests(plan, frozen)
    options = replace(
        CollectionOptions.from_mapping(exported["collection"]),
        grader=plan["judge"]["requested_model"],
        requests_per_minute=10000,
    )
    runs = {
        f"{side}_run": json.loads((FIXTURES / f"{side}_run.json").read_text())
        for side in ("baseline", "subject")
    }
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-2024-08-06",
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
