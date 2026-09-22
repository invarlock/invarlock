"""Adversarial collector and replay tests for compatible judge services."""

from __future__ import annotations

import asyncio
import gzip
import json
import threading
import time
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from invarlock.judge_measurements.contracts import (
    JudgeMeasurementContractError,
    _check_openai_compatible_sources,
    _openai_compatible_source_errors,
    canonical_payload,
    render_judge_request,
    validate_measurement_plan,
    validate_measurements,
)
from invarlock.judge_measurements.openai_compatible import (
    OpenAICompatibleJudgeError,
    OpenAICompatibleJudgeOptions,
    _bounded_exchange,
    _checkpoint_sources,
    _checkpoint_sources_pinned,
    _client,
    _exchange_once,
    _parse_rating,
    _request_body,
    _safe_headers,
    _validate_cached,
    collect_openai_compatible,
    openai_compatible_service_identity,
    preflight_openai_compatible,
    validate_openai_compatible_collection,
)
from invarlock.judge_measurements.openai_compatible_contract import (
    OpenAICompatibleContractError,
    contains_secret,
    decode_http_blob,
    failure_details,
    has_credential_field,
    normalize_configuration,
    normalized_base_url,
    response_facts,
)
from invarlock.security import temporarily_allow_network
from tests.judge_measurements.test_openai_compatible_workflows import (
    configuration,
    stage,
)


def _inputs(tmp_path: Path) -> tuple[dict[str, Any], ...]:
    request = stage(tmp_path, "v3")
    return tuple(
        json.loads(request.inputs[name].read_bytes())
        for name in ("plan", "collection", "baseline_run", "subject_run")
    )


def _success_payload(**updates: Any) -> bytes:
    value = {
        "id": "response-1",
        "model": "test/judge",
        "system_fingerprint": "fp-1",
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
        "usage": {"prompt_tokens": 20, "completion_tokens": 4},
    }
    value.update(updates)
    return canonical_payload(value)


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: int = 200,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    error: BaseException | None = None,
) -> list[bytes]:
    from invarlock.judge_measurements import openai_compatible

    calls: list[bytes] = []
    payload = _success_payload() if body is None else body
    response_headers = {"content-type": "application/json", **(headers or {})}

    class Stream:
        async def __aenter__(self):
            if error is not None:
                raise error

            async def chunks(chunk_size: int):
                assert chunk_size == 64 * 1024
                served = payload
                try:
                    decoded = json.loads(payload)
                except (UnicodeError, ValueError):
                    pass
                else:
                    if decoded.get("id") == "response-1":
                        decoded["id"] = f"response-{len(calls)}"
                        served = canonical_payload(decoded)
                yield served

            return SimpleNamespace(
                status_code=status,
                headers=response_headers,
                aiter_raw=chunks,
            )

        async def __aexit__(self, *_: object) -> bool:
            return False

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> bool:
            return False

        def stream(self, method: str, path: str, *, content: bytes):
            assert (method, path) == ("POST", "chat/completions")
            calls.append(content)
            return Stream()

    def client(config: dict[str, Any], request_headers: dict[str, str]):
        assert request_headers["accept-encoding"] == "identity"
        return Client()

    monkeypatch.setattr(openai_compatible, "_client", client)
    monkeypatch.setenv("INVARLOCK_ALLOW_JUDGE_NETWORK", "1")
    return calls


def _collect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **transport: Any,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], list[bytes]]:
    plan, collection, baseline, subject = _inputs(tmp_path)
    calls = _install_transport(monkeypatch, **transport)
    result = collect_openai_compatible(
        plan,
        configuration=collection,
        baseline_run=baseline,
        subject_run=subject,
        options=OpenAICompatibleJudgeOptions(
            checkpoint_directory=tmp_path / "checkpoint"
        ),
    )
    return result, (plan, collection, baseline, subject), calls


@pytest.mark.parametrize(
    "base_url",
    [
        "http://user:pass@localhost:8000/v1",
        "http://localhost:8000/v1?key=value",
        "http://localhost:8000/v1#fragment",
        "http://localhost:bad/v1",
        "http://localhost:8000/a/../v1",
        "http://localhost:8000/a//v1",
        "http://localhost:8000/%76%31",
        "http://localhost:8000/v1\\x",
        "http://localhost:8000/v1\x00",
        "https://api.openai.com./v1",
    ],
)
def test_service_identity_rejects_unsafe_or_hosted_endpoints(base_url):
    value = configuration()
    value["base_url"] = base_url
    with pytest.raises(OpenAICompatibleJudgeError):
        openai_compatible_service_identity(value)


def test_service_identity_normalizes_default_port_and_dns_case():
    first = configuration()
    first["base_url"] = "HTTPS://LOCALHOST.:443/api/v1"
    second = configuration()
    second["base_url"] = "https://localhost/api/v1/"
    assert openai_compatible_service_identity(
        first
    ) == openai_compatible_service_identity(second)


def test_remote_plaintext_bearer_is_rejected_but_loopback_is_allowed():
    remote = configuration()
    remote.update(base_url="http://example.test/v1", authentication="bearer_env")
    with pytest.raises(OpenAICompatibleJudgeError, match="requires HTTPS"):
        openai_compatible_service_identity(remote)
    local = configuration()
    local["authentication"] = "bearer_env"
    assert openai_compatible_service_identity(local)["service"] == "vllm"


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "ftp://localhost/v1",
        "http://[fe80::1%25eth0]/v1",
        "http://./v1",
        "http://localhost/v1\x80",
        "http://localhost/v1 ",
        "http://localhost:65536/v1",
        "http://\ud800.test/v1",
    ],
)
def test_pure_url_contract_rejects_hostile_types_and_encodings(value):
    with pytest.raises(OpenAICompatibleContractError):
        normalized_base_url(value)


def test_pure_url_contract_canonicalizes_ipv6():
    assert normalized_base_url("HTTP://[0:0:0:0:0:0:0:1]:80/v1") == ("http://[::1]/v1/")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", "other"),
        ("service", []),
        ("service", "other"),
        ("model", []),
        ("model", ""),
        ("model", "bad model"),
        ("authentication", []),
        ("authentication", "other"),
        ("response_format", []),
        ("response_format", "other"),
        ("request_timeout_seconds", True),
        ("request_timeout_seconds", 0),
        ("max_calls", 200_001),
        ("max_input_bytes", 0),
        ("max_output_tokens", 10**12 + 1),
    ],
)
def test_pure_configuration_contract_rejects_invalid_fields(field, value):
    config = configuration()
    config[field] = value
    with pytest.raises(OpenAICompatibleContractError):
        normalize_configuration(config, maximum_input_bytes=384 * 1024 * 1024)


def test_pure_configuration_contract_rejects_non_mapping_and_unknown_field():
    with pytest.raises(OpenAICompatibleContractError):
        normalize_configuration([], maximum_input_bytes=10)
    config = configuration()
    config["unknown"] = True
    with pytest.raises(OpenAICompatibleContractError):
        normalize_configuration(config, maximum_input_bytes=384 * 1024 * 1024)


def test_recursive_secret_and_credential_detection():
    assert has_credential_field({"nested": [{"API_KEY": "value"}]})
    assert not has_credential_field({"nested": ["value"]})
    assert contains_secret({"prefix-sk-test": None}, "sk-test")
    assert contains_secret({"nested": ["prefix-sk-test"]}, "sk-test")
    assert not contains_secret({"nested": [1]}, "sk-test")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: [],
        lambda value: {**value, "authorization": "x"},
        lambda value: {**value, "choices": []},
        lambda value: {**value, "choices": [None]},
        lambda value: {
            **value,
            "choices": [{**value["choices"][0], "message": None}],
        },
        lambda value: {
            **value,
            "choices": [
                {
                    **value["choices"][0],
                    "message": {"role": "user", "content": "{}"},
                }
            ],
        },
        lambda value: {**value, "model": []},
        lambda value: {**value, "model": ""},
        lambda value: {**value, "model": "bad model"},
        lambda value: {
            **value,
            "choices": [{**value["choices"][0], "index": True}],
        },
        lambda value: {
            **value,
            "choices": [
                {
                    **value["choices"][0],
                    "message": {
                        **value["choices"][0]["message"],
                        "tool_calls": [{"id": "x"}],
                    },
                }
            ],
        },
        lambda value: {
            **value,
            "choices": [{**value["choices"][0], "finish_reason": "length"}],
        },
        lambda value: {**value, "id": []},
        lambda value: {**value, "system_fingerprint": "x" * 257},
        lambda value: {**value, "usage": []},
        lambda value: {
            **value,
            "usage": {"prompt_tokens": True, "completion_tokens": 1},
        },
    ],
)
def test_pure_response_projection_rejects_hostile_shapes(mutation):
    value = json.loads(_success_payload())
    with pytest.raises(OpenAICompatibleContractError):
        response_facts(mutation(value), approved_models=["test/judge"])


def test_pure_response_projection_handles_optional_usage_and_model_approval():
    value = json.loads(_success_payload())
    value.pop("usage")
    assert response_facts(value, approved_models=["test/judge"])["usage"] is None
    with pytest.raises(OpenAICompatibleContractError, match="not approved"):
        response_facts(value, approved_models=["other"])


@pytest.mark.parametrize(
    "blob",
    [
        None,
        {},
        {"media_type": [], "encoding": "utf-8", "text": "x", "sha256": "x"},
        {"media_type": "x", "encoding": [], "text": "x", "sha256": "x"},
        {"media_type": "x", "encoding": "other", "text": "x", "sha256": "x"},
        {"media_type": "x", "encoding": "base64", "text": "!", "sha256": "x"},
        {"media_type": "x", "encoding": "utf-8", "text": "x", "sha256": "0" * 64},
    ],
)
def test_http_blob_decoder_rejects_hostile_shapes(blob):
    with pytest.raises(OpenAICompatibleContractError):
        decode_http_blob(blob)


def test_http_blob_decoder_accepts_exact_base64():
    import base64
    import hashlib

    payload = b"\xff"
    assert (
        decode_http_blob(
            {
                "media_type": "application/octet-stream",
                "encoding": "base64",
                "text": base64.b64encode(payload).decode(),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        == payload
    )


@pytest.mark.parametrize(
    "options",
    [
        OpenAICompatibleJudgeOptions(scorer_id=""),
        OpenAICompatibleJudgeOptions(scorer_id="bad\x00id"),
        OpenAICompatibleJudgeOptions(checkpoint_directory="bad"),
    ],
)
def test_collection_options_reject_invalid_public_values(options):
    with pytest.raises(OpenAICompatibleJudgeError):
        options.validate()
    with pytest.raises(OpenAICompatibleContractError):
        failure_details("unknown", None)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["judge"].update(requested_model="other"),
        lambda plan: plan["judge"]["config"].update(reasoning_effort="low"),
        lambda plan: plan["schedule"].update(
            max_attempts=2, retry_on=["transport_error"]
        ),
    ],
)
def test_endpoint_plan_constraints_reject_misbound_or_retrying_plans(
    tmp_path, mutation
):
    plan, collection, *_ = _inputs(tmp_path)
    mutation(plan)
    with pytest.raises((OpenAICompatibleJudgeError, JudgeMeasurementContractError)):
        validate_openai_compatible_collection(collection, plan)


@pytest.mark.parametrize("field", ["max_calls", "max_output_tokens"])
def test_endpoint_budgets_must_reserve_the_complete_plan(tmp_path, field):
    plan, collection, *_ = _inputs(tmp_path)
    collection[field] = 1
    with pytest.raises(OpenAICompatibleJudgeError, match="reserve every"):
        validate_openai_compatible_collection(collection, plan)


def test_endpoint_plan_rejects_source_count_before_rendering(tmp_path, monkeypatch):
    from invarlock.judge_measurements import openai_compatible

    plan, collection, baseline, subject = _inputs(tmp_path)
    binding = plan["answer_bindings"][0]
    plan["answer_bindings"] = [
        {**binding, "case_id": f"case-{index:03d}"} for index in range(101)
    ]
    plan["sampling"]["case_units"] = [
        {"case_id": item["case_id"], "unit_id": item["case_id"]}
        for item in plan["answer_bindings"]
    ]
    plan["schedule"].update(repetitions=5, expected_trials=1010)
    collection.update(max_calls=1010, max_output_tokens=1010 * 128)
    with pytest.raises(OpenAICompatibleJudgeError, match="source-count limit"):
        validate_openai_compatible_collection(collection, plan)

    monkeypatch.setattr(
        openai_compatible,
        "render_judge_request",
        lambda *_args, **_kwargs: pytest.fail("oversized plan rendered a request"),
    )
    with pytest.raises(OpenAICompatibleJudgeError, match="source-count limit"):
        collect_openai_compatible(
            plan,
            configuration=collection,
            baseline_run=baseline,
            subject_run=subject,
        )


def test_endpoint_input_budget_stops_at_first_exceeding_request(tmp_path, monkeypatch):
    from invarlock.judge_measurements import openai_compatible

    plan, collection, baseline, subject = _inputs(tmp_path)
    collection["max_input_bytes"] = 1
    monkeypatch.setenv("INVARLOCK_ALLOW_JUDGE_NETWORK", "1")
    rendered = 0
    original = openai_compatible.render_judge_request

    def counted_render(*args, **kwargs):
        nonlocal rendered
        rendered += 1
        if rendered > 1:
            pytest.fail("collector rendered after exhausting the input-byte budget")
        return original(*args, **kwargs)

    monkeypatch.setattr(openai_compatible, "render_judge_request", counted_render)
    monkeypatch.setattr(
        openai_compatible,
        "_client",
        lambda *_args, **_kwargs: pytest.fail("input budget admitted a network call"),
    )
    with pytest.raises(OpenAICompatibleJudgeError, match="input-byte reservation"):
        collect_openai_compatible(
            plan,
            configuration=collection,
            baseline_run=baseline,
            subject_run=subject,
        )
    assert rendered == 1


def test_explicit_json_schema_transport_is_plan_bound_and_replayed(
    tmp_path, monkeypatch
):
    plan, collection, baseline, subject = _inputs(tmp_path)
    collection.update(service="lm_studio", response_format="json_schema")
    identity = openai_compatible_service_identity(collection)
    assert identity["response_format"] == "json_schema"
    plan["judge"]["service_identity"] = identity
    calls = _install_transport(monkeypatch)
    measurements = collect_openai_compatible(
        plan,
        configuration=collection,
        baseline_run=baseline,
        subject_run=subject,
        options=OpenAICompatibleJudgeOptions(
            checkpoint_directory=tmp_path / "checkpoint"
        ),
    )
    expected_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rating": {
                "type": "string",
                "enum": [item["label"] for item in plan["scale"]["ratings"]],
            }
        },
        "required": ["rating"],
    }
    assert len(calls) == 4
    assert all(
        json.loads(body)["response_format"]
        == {
            "type": "json_schema",
            "json_schema": {
                "name": "invarlock_judge_rating",
                "strict": True,
                "schema": expected_schema,
            },
        }
        for body in calls
    )
    assert measurements["completeness"]["completed_trials"] == 4

    unapproved = {**collection, "response_format": "json_object"}
    with pytest.raises(OpenAICompatibleJudgeError, match="identity differs"):
        validate_openai_compatible_collection(unapproved, plan)


def test_absent_response_format_preserves_json_object_wire_contract(
    tmp_path, monkeypatch
):
    measurements, _, calls = _collect(tmp_path, monkeypatch)
    assert all(
        json.loads(body)["response_format"] == {"type": "json_object"} for body in calls
    )
    assert all(
        "response_format" not in json.loads(source["content"])["collection"]
        for source in measurements["sources"]
    )


def test_preflight_rejects_missing_wrong_dependency_and_credential(
    tmp_path, monkeypatch
):
    from invarlock.judge_measurements import openai_compatible

    plan, collection, *_ = _inputs(tmp_path)
    monkeypatch.setattr(
        openai_compatible.importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(
            openai_compatible.importlib.metadata.PackageNotFoundError
        ),
    )
    with pytest.raises(OpenAICompatibleJudgeError, match="httpx==0.28.1"):
        preflight_openai_compatible(collection, plan)
    monkeypatch.setattr(
        openai_compatible.importlib.metadata, "version", lambda _name: "0"
    )
    with pytest.raises(OpenAICompatibleJudgeError, match="httpx==0.28.1"):
        preflight_openai_compatible(collection, plan)
    monkeypatch.setattr(
        openai_compatible.importlib.metadata, "version", lambda _name: "0.28.1"
    )
    collection["authentication"] = "bearer_env"
    monkeypatch.delenv("INVARLOCK_OPENAI_COMPATIBLE_API_KEY", raising=False)
    with pytest.raises(OpenAICompatibleJudgeError, match="must be set"):
        preflight_openai_compatible(collection, plan)


def test_header_and_rating_helpers_cover_closed_failure_shapes(tmp_path):
    class Headers:
        def __init__(self, value):
            self.value = value

        def get(self, name):
            return self.value if name == "server" else None

    assert _safe_headers(SimpleNamespace(headers=Headers(7)), None) == {"server": "7"}
    with pytest.raises(OpenAICompatibleJudgeError, match="bearer credential"):
        _safe_headers(SimpleNamespace(headers=Headers("echo-secret")), "secret")
    with pytest.raises(OpenAICompatibleJudgeError, match="invalid retained header"):
        _safe_headers(SimpleNamespace(headers=Headers("x" * 513)), None)
    plan, *_ = _inputs(tmp_path)
    assert _parse_rating("not-json", plan)["status"] == "invalid"
    assert _parse_rating('{"rating":"unknown"}', plan)["status"] == "invalid"
    plan["judge"]["config"]["seed"] = None
    request = render_judge_request(
        plan, input_text="input", answer_text="answer", reference_text=None
    )
    with pytest.raises(OpenAICompatibleJudgeError, match="explicit seed"):
        _request_body(plan, request, "test/judge", "json_object")


def test_checkpoint_parser_rejects_every_ambiguous_layout(tmp_path):
    options = OpenAICompatibleJudgeOptions(checkpoint_directory=None)
    assert _checkpoint_sources(options, b"identity", 1) == []

    mismatch = tmp_path / "mismatch"
    mismatch.mkdir(mode=0o700)
    assert _checkpoint_sources_pinned(mismatch, b"one", 1) == []
    with pytest.raises(OpenAICompatibleJudgeError, match="different collection"):
        _checkpoint_sources_pinned(mismatch, b"two", 1)

    unexpected = tmp_path / "unexpected"
    unexpected.mkdir(mode=0o700)
    (unexpected / "surprise").write_text("x")
    with pytest.raises(OpenAICompatibleJudgeError, match="unexpected entry"):
        _checkpoint_sources_pinned(unexpected, b"identity", 1)

    missing_admission = tmp_path / "missing-admission"
    missing_admission.mkdir(mode=0o700)
    _checkpoint_sources_pinned(missing_admission, b"identity", 1)
    (missing_admission / "result-000001.json").write_text("{}")
    with pytest.raises(OpenAICompatibleJudgeError, match="lacks admission"):
        _checkpoint_sources_pinned(missing_admission, b"identity", 1)

    bad_admission = tmp_path / "bad-admission"
    bad_admission.mkdir(mode=0o700)
    _checkpoint_sources_pinned(bad_admission, b"identity", 1)
    (bad_admission / "admission-000001.json").write_text("{}")
    (bad_admission / "result-000001.json").write_text("{}")
    with pytest.raises(OpenAICompatibleJudgeError, match="admission is invalid"):
        _checkpoint_sources_pinned(bad_admission, b"identity", 1)

    noncanonical = tmp_path / "noncanonical"
    noncanonical.mkdir(mode=0o700)
    _checkpoint_sources_pinned(noncanonical, b"identity", 1)
    (noncanonical / "admission-000001.json").write_bytes(
        canonical_payload(
            {
                "format": "invarlock/openai-compatible-judge-admission-v1",
                "batch": 1,
            }
        )
    )
    (noncanonical / "result-000001.json").write_text('{ "value": 1 }')
    with pytest.raises(OpenAICompatibleJudgeError, match="canonical object"):
        _checkpoint_sources_pinned(noncanonical, b"identity", 1)

    gap = tmp_path / "gap"
    gap.mkdir(mode=0o700)
    _checkpoint_sources_pinned(gap, b"identity", 2)
    (gap / "admission-000002.json").write_text("{}")
    with pytest.raises(OpenAICompatibleJudgeError, match="non-contiguous"):
        _checkpoint_sources_pinned(gap, b"identity", 2)


@pytest.mark.parametrize(
    ("transport", "outcome", "body_retained"),
    [
        ({"error": OSError("private details")}, "transport_error", False),
        ({"status": 503, "body": b"temporarily unavailable"}, "http_error", True),
        ({"body": b"not json"}, "malformed_response", True),
        ({"body": b"\xff\xfe"}, "malformed_response", True),
        (
            {"body": _success_payload(model="unapproved/model")},
            "malformed_response",
            True,
        ),
        (
            {"headers": {"content-encoding": "gzip"}, "body": gzip.compress(b"{}")},
            "malformed_response",
            False,
        ),
        (
            {"headers": {"server": "x" * 513}},
            "malformed_response",
            False,
        ),
        ({"body": b"x" * (2 * 1024 * 1024 + 1)}, "response_too_large", False),
    ],
)
def test_failures_are_durable_redacted_and_replayable(
    tmp_path, monkeypatch, transport, outcome, body_retained
):
    measurements, _, calls = _collect(tmp_path, monkeypatch, **transport)
    assert len(calls) == 4
    assert measurements["completeness"] == {
        "status": "incomplete",
        "expected_trials": 4,
        "recorded_trials": 4,
        "completed_trials": 0,
    }
    for retained in measurements["sources"]:
        source = json.loads(retained["content"])
        assert source["http"]["outcome"] == outcome
        assert (source["http"]["response_body"] is not None) is body_retained
        assert source["trials"][0]["attempts"][0]["status"] == "cancelled"


@pytest.mark.parametrize("location", ["body", "header", "json_escaped", "json_error"])
def test_bearer_echo_is_never_retained(tmp_path, monkeypatch, location):
    secret = "sk-test"
    plan, collection, baseline, subject = _inputs(tmp_path)
    collection["authentication"] = "bearer_env"
    monkeypatch.setenv("INVARLOCK_OPENAI_COMPATIBLE_API_KEY", secret)
    if location == "header":
        transport = {"headers": {"x-request-id": f"echo-{secret}"}}
    elif location in {"json_escaped", "json_error"}:
        source = (
            canonical_payload({"error": secret})
            if location == "json_error"
            else _success_payload(id=secret)
        )
        transport = {
            "body": source.replace(secret.encode(), b"sk\\u002dtest"),
            **({"status": 400} if location == "json_error" else {}),
        }
    else:
        transport = {"body": _success_payload(extra=secret)}
    _install_transport(monkeypatch, **transport)
    measurements = collect_openai_compatible(
        plan,
        configuration=collection,
        baseline_run=baseline,
        subject_run=subject,
        options=OpenAICompatibleJudgeOptions(
            checkpoint_directory=tmp_path / "checkpoint"
        ),
    )
    encoded = canonical_payload(measurements)
    assert secret.encode() not in encoded
    assert all(
        json.loads(item["content"])["http"]["outcome"] == "credential_echo"
        for item in measurements["sources"]
    )


@pytest.mark.parametrize(
    "body",
    [b'{"error":"sk\\u002dtest"', b"\xff\xfe"],
)
def test_bearer_malformed_bodies_are_redacted_without_overclaiming_echo(
    tmp_path, monkeypatch, body
):
    secret = "sk-test"
    plan, collection, baseline, subject = _inputs(tmp_path)
    collection["authentication"] = "bearer_env"
    monkeypatch.setenv("INVARLOCK_OPENAI_COMPATIBLE_API_KEY", secret)
    _install_transport(monkeypatch, status=400, body=body)
    measurements = collect_openai_compatible(
        plan,
        configuration=collection,
        baseline_run=baseline,
        subject_run=subject,
        options=OpenAICompatibleJudgeOptions(
            checkpoint_directory=tmp_path / "checkpoint"
        ),
    )
    assert secret.encode() not in canonical_payload(measurements)
    for retained in measurements["sources"]:
        source = json.loads(retained["content"])
        assert source["http"]["outcome"] == "malformed_response"
        assert source["http"]["response_body"] is None


def _rehash_source(
    measurements: dict[str, Any], source: dict[str, Any], index: int = 0
) -> None:
    payload = canonical_payload(source)
    retained = measurements["sources"][index]
    retained.update(
        content=payload.decode("utf-8"),
        byte_size=len(payload),
        sha256=__import__("hashlib").sha256(payload).hexdigest(),
    )


def _response_blob(payload: bytes) -> dict[str, Any]:
    return {
        "media_type": "application/json",
        "encoding": "utf-8",
        "text": payload.decode(),
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
    }


def test_pure_replay_rejects_closed_endpoint_source_failure_shapes(
    tmp_path, monkeypatch
):
    success_root = tmp_path / "success"
    success_root.mkdir()
    measurements, artifacts, _ = _collect(success_root, monkeypatch)
    plan = artifacts[0]
    success = json.loads(measurements["sources"][0]["content"])

    malformed = deepcopy(success)
    malformed.pop("http")
    assert "unsupported shape" in _openai_compatible_source_errors(malformed, plan)[0]

    invalid_configuration = deepcopy(success)
    invalid_configuration["collection"].pop("service")
    assert (
        "configuration is invalid"
        in _openai_compatible_source_errors(invalid_configuration, plan)[0]
    )

    noncanonical = deepcopy(success)
    noncanonical["collection"]["base_url"] = noncanonical["collection"][
        "base_url"
    ].rstrip("/")
    assert any(
        "endpoint is not canonical" in error
        for error in _openai_compatible_source_errors(noncanonical, plan)
    )

    no_attempt = deepcopy(success)
    no_attempt["trials"][0]["attempts"] = []
    assert any(
        "must contain one attempt" in error
        for error in _openai_compatible_source_errors(no_attempt, plan)
    )

    invalid_request = deepcopy(success)
    invalid_request["trials"][0]["attempts"][0]["request"] = None
    assert any(
        "request material is invalid" in error
        for error in _openai_compatible_source_errors(invalid_request, plan)
    )

    malformed_request = deepcopy(success)
    malformed_request["trials"][0]["attempts"][0]["request"]["text"] = "{"
    assert (
        "request material is invalid"
        in _openai_compatible_source_errors(malformed_request, plan)[0]
    )

    non_object_request = deepcopy(success)
    non_object_request["trials"][0]["attempts"][0]["request"]["text"] = "[]"
    assert (
        "must be an object"
        in _openai_compatible_source_errors(non_object_request, plan)[0]
    )

    noncanonical_wire = deepcopy(success)
    noncanonical_wire["http"]["request"] = {"nonfinite": float("nan")}
    assert any(
        "wire request differs" in error
        for error in _openai_compatible_source_errors(noncanonical_wire, plan)
    )

    invalid_headers = deepcopy(success)
    invalid_headers["http"]["response_headers"] = []
    assert any(
        "response headers are invalid" in error
        for error in _openai_compatible_source_errors(invalid_headers, plan)
    )

    invalid_status = deepcopy(success)
    invalid_status["http"]["response_status"] = None
    assert any(
        "HTTP status is invalid" in error
        for error in _openai_compatible_source_errors(invalid_status, plan)
    )

    oversized = deepcopy(success)
    oversized["http"]["response_body"] = _response_blob(b"x" * (2 * 1024 * 1024 + 1))
    assert any(
        "exceeds its byte limit" in error
        for error in _openai_compatible_source_errors(oversized, plan)
    )

    credential_body = deepcopy(success)
    credential_body["http"]["response_body"] = _response_blob(
        b'{"authorization":"secret"}'
    )
    assert any(
        "contains credential material" in error
        for error in _openai_compatible_source_errors(credential_body, plan)
    )

    incomplete_success = deepcopy(success)
    incomplete_success["http"]["response_body"] = None
    assert any(
        "success response is incomplete" in error
        for error in _openai_compatible_source_errors(incomplete_success, plan)
    )

    invalid_success = deepcopy(success)
    invalid_success["http"]["response_body"] = _response_blob(b"{}")
    assert any(
        "success response is invalid" in error
        for error in _openai_compatible_source_errors(invalid_success, plan)
    )

    failure_root = tmp_path / "failure"
    failure_root.mkdir()
    failed_measurements, _, _ = _collect(
        failure_root, monkeypatch, status=400, body=b'{"error":"safe"}'
    )
    failure = json.loads(failed_measurements["sources"][0]["content"])

    invalid_http_error = deepcopy(failure)
    invalid_http_error["http"]["response_status"] = 200
    assert any(
        "HTTP error outcome is invalid" in error
        for error in _openai_compatible_source_errors(invalid_http_error, plan)
    )

    unsafe_body = deepcopy(failure)
    unsafe_body["http"]["outcome"] = "response_too_large"
    assert any(
        "unsafe failure retained a response body" in error
        for error in _openai_compatible_source_errors(unsafe_body, plan)
    )

    unsafe_headers = deepcopy(failure)
    unsafe_headers["http"].update(
        outcome="transport_error", response_status=None, response_body=None
    )
    assert any(
        "unsafe failure retained response headers" in error
        for error in _openai_compatible_source_errors(unsafe_headers, plan)
    )

    invalid_failure = deepcopy(failure)
    invalid_failure["service_identity"]["response_model"] = "invented"
    assert any(
        "failed outcome is invalid" in error
        for error in _openai_compatible_source_errors(invalid_failure, plan)
    )


def test_pure_replay_rejects_inconsistent_endpoint_shards_and_budgets(
    tmp_path, monkeypatch
):
    measurements, artifacts, _ = _collect(tmp_path, monkeypatch)
    plan = artifacts[0]
    sources = [
        json.loads(measurements["sources"][index]["content"]) for index in range(2)
    ]
    inconsistent = deepcopy(sources)
    inconsistent[1]["collection"]["max_calls"] += 1
    with pytest.raises(
        JudgeMeasurementContractError, match="inconsistent endpoint configuration"
    ):
        _check_openai_compatible_sources(inconsistent, plan)

    exhausted = deepcopy(sources[:1])
    exhausted[0]["collection"]["max_input_bytes"] = 1
    with pytest.raises(
        JudgeMeasurementContractError, match="aggregate resource reservations"
    ):
        _check_openai_compatible_sources(exhausted, plan)


@pytest.mark.parametrize(
    "tamper",
    [
        "bool_n",
        "usage",
        "finish",
        "endpoint",
        "body",
        "outcome_shape",
        "encoding_shape",
    ],
)
def test_replay_rejects_forged_sources_after_hash_recomputation(
    tmp_path, monkeypatch, tamper
):
    measurements, artifacts, _ = _collect(tmp_path, monkeypatch)
    plan, _, baseline, subject = artifacts
    forged = deepcopy(measurements)
    source = json.loads(forged["sources"][0]["content"])
    if tamper == "bool_n":
        source["http"]["request"]["n"] = True
    elif tamper == "usage":
        source["trials"][0]["attempts"][0]["usage"]["input_tokens"] += 1
        forged["trials"][0] = deepcopy(source["trials"][0])
    elif tamper == "finish":
        source["trials"][0]["attempts"][0]["finish_reason"] = "length"
        forged["trials"][0] = deepcopy(source["trials"][0])
    elif tamper == "endpoint":
        source["collection"]["base_url"] = "http://127.0.0.1:9000/v1/"
    elif tamper == "body":
        source["http"]["response_body"]["sha256"] = "0" * 64
    elif tamper == "outcome_shape":
        source["http"]["outcome"] = []
    else:
        source["http"]["response_body"]["encoding"] = []
    _rehash_source(forged, source)
    with pytest.raises(JudgeMeasurementContractError):
        validate_measurements(
            forged,
            plan,
            baseline_run=baseline,
            subject_run=subject,
        )


def test_replay_rejects_repeated_non_null_compatible_response_ids(
    tmp_path, monkeypatch
):
    import hashlib

    measurements, artifacts, _ = _collect(tmp_path, monkeypatch)
    plan, _, baseline, subject = artifacts
    forged = deepcopy(measurements)
    first = json.loads(forged["sources"][0]["content"])
    second = json.loads(forged["sources"][1]["content"])
    repeated = first["service_identity"]["request_id"]
    response_body = second["http"]["response_body"]
    response = json.loads(response_body["text"])
    response["id"] = repeated
    raw = canonical_payload(response)
    response_body.update(text=raw.decode(), sha256=hashlib.sha256(raw).hexdigest())
    second["service_identity"]["request_id"] = repeated
    second["trials"][0]["attempts"][0]["request_id"] = repeated
    forged["trials"][1] = deepcopy(second["trials"][0])
    _rehash_source(forged, second, 1)
    with pytest.raises(
        JudgeMeasurementContractError, match="response IDs must be unique"
    ):
        validate_measurements(
            forged,
            plan,
            baseline_run=baseline,
            subject_run=subject,
        )


def test_checkpoint_resume_uses_zero_network_calls(tmp_path, monkeypatch):
    measurements, artifacts, calls = _collect(tmp_path, monkeypatch)
    assert len(calls) == 4
    plan, collection, baseline, subject = artifacts
    from invarlock.judge_measurements import openai_compatible

    monkeypatch.setattr(
        openai_compatible,
        "_client",
        lambda *_args, **_kwargs: pytest.fail("resume contacted the service"),
    )
    resumed = collect_openai_compatible(
        plan,
        configuration=collection,
        baseline_run=baseline,
        subject_run=subject,
        options=OpenAICompatibleJudgeOptions(
            checkpoint_directory=tmp_path / "checkpoint"
        ),
    )
    assert resumed == measurements


@pytest.mark.parametrize(
    "fault", ["source_error", "configuration", "trials", "slot", "mapping"]
)
def test_cached_source_validation_rejects_each_substitution(
    tmp_path, monkeypatch, fault
):
    from invarlock.judge_measurements import openai_compatible

    measurements, artifacts, _ = _collect(tmp_path, monkeypatch)
    plan, collection, *_ = artifacts
    source = json.loads(measurements["sources"][0]["content"])
    trial = source["trials"][0]
    expected = {
        name: trial[name]
        for name in (
            "trial_id",
            "case_id",
            "side",
            "repetition",
            "answer_sha256",
            "plan_sha256",
        )
    }
    request = trial["attempts"][0]["request"]["text"].encode()
    pending = [(expected, request, source["http"]["request"])] * 4
    if fault == "source_error":
        monkeypatch.setattr(
            openai_compatible,
            "_openai_compatible_source_errors",
            lambda *_args: ["forged source"],
        )
    else:
        monkeypatch.setattr(
            openai_compatible,
            "_openai_compatible_source_errors",
            lambda *_args: [],
        )
        if fault == "configuration":
            source["collection"] = {**collection, "max_calls": 5}
        elif fault == "trials":
            source["trials"] = []
        elif fault == "slot":
            source["trials"][0]["case_id"] = "other"
        else:
            source["trials"][0]["attempts"][0]["source"]["scorer_id"] = "other"
    with pytest.raises(OpenAICompatibleJudgeError):
        _validate_cached(
            [source],
            pending=pending,
            plan=plan,
            configuration=collection,
            options=OpenAICompatibleJudgeOptions(
                scorer_id="judge", checkpoint_directory=tmp_path / "checkpoint"
            ),
        )


def test_cached_sources_reject_repeated_non_null_response_ids(tmp_path, monkeypatch):
    from invarlock.judge_measurements import openai_compatible

    plan, collection, *_ = _inputs(tmp_path)
    monkeypatch.setattr(
        openai_compatible, "_openai_compatible_source_errors", lambda *_args: []
    )
    monkeypatch.setattr(
        openai_compatible, "_validate_measurement_trial_shape", lambda *_: None
    )
    monkeypatch.setattr(
        openai_compatible, "_check_attempts", lambda *_args, **_kwargs: None
    )
    pending = []
    sources = []
    for index in range(2):
        trial = {
            "trial_id": f"trial-{index}",
            "case_id": f"case-{index}",
            "side": "baseline",
            "repetition": 1,
            "answer_sha256": "0" * 64,
            "plan_sha256": "1" * 64,
            "attempts": [
                {
                    "request_id": "repeated-id",
                    "source": {
                        "source_id": f"openai-compatible-judge-{index + 1:06d}",
                        "scorer_id": "judge",
                        "model_event_id": f"trial-{index}",
                        "record_index": 0,
                        "attempt_index": 0,
                    },
                }
            ],
        }
        expected = {
            name: trial[name]
            for name in (
                "trial_id",
                "case_id",
                "side",
                "repetition",
                "answer_sha256",
                "plan_sha256",
            )
        }
        pending.append((expected, b"{}", {}))
        sources.append({"collection": collection, "trials": [trial]})
    with pytest.raises(
        OpenAICompatibleJudgeError, match="reused a non-null response ID"
    ):
        _validate_cached(
            sources,
            pending=pending,
            plan=plan,
            configuration=collection,
            options=OpenAICompatibleJudgeOptions(scorer_id="judge"),
        )


def test_small_collection_can_run_without_checkpoint(tmp_path, monkeypatch):
    plan, collection, baseline, subject = _inputs(tmp_path)
    calls = _install_transport(monkeypatch)
    measurements = collect_openai_compatible(
        plan,
        configuration=collection,
        baseline_run=baseline,
        subject_run=subject,
    )
    assert measurements["completeness"]["completed_trials"] == 4
    assert len(calls) == 4


def test_collection_rejects_network_input_and_client_before_admission(
    tmp_path, monkeypatch
):
    from invarlock.judge_measurements import openai_compatible

    plan, collection, baseline, subject = _inputs(tmp_path)
    monkeypatch.delenv("INVARLOCK_ALLOW_JUDGE_NETWORK", raising=False)
    monkeypatch.setattr(openai_compatible, "network_policy_allows", lambda: False)
    with pytest.raises(OpenAICompatibleJudgeError, match="allowed network policy"):
        collect_openai_compatible(
            plan,
            configuration=collection,
            baseline_run=baseline,
            subject_run=subject,
        )

    monkeypatch.setenv("INVARLOCK_ALLOW_JUDGE_NETWORK", "1")
    too_small = dict(collection)
    too_small["max_input_bytes"] = 1
    with pytest.raises(OpenAICompatibleJudgeError, match="input-byte reservation"):
        collect_openai_compatible(
            plan,
            configuration=too_small,
            baseline_run=baseline,
            subject_run=subject,
        )

    monkeypatch.setattr(
        openai_compatible,
        "_client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private")),
    )
    checkpoint = tmp_path / "client-checkpoint"
    with pytest.raises(OpenAICompatibleJudgeError, match="could not be initialized"):
        collect_openai_compatible(
            plan,
            configuration=collection,
            baseline_run=baseline,
            subject_run=subject,
            options=OpenAICompatibleJudgeOptions(checkpoint_directory=checkpoint),
        )
    assert not list(checkpoint.glob("admission-*.json"))


def test_collection_capacity_and_credential_defenses_run_before_calls(
    tmp_path, monkeypatch
):
    from invarlock.judge_measurements import openai_compatible

    plan, collection, baseline, subject = _inputs(tmp_path)
    monkeypatch.setenv("INVARLOCK_ALLOW_JUDGE_NETWORK", "1")
    monkeypatch.setattr(
        openai_compatible,
        "_NEXT_SOURCE_RESERVATION_BYTES",
        openai_compatible.MEASUREMENTS_MAX_BYTES,
    )
    with pytest.raises(OpenAICompatibleJudgeError, match="durable checkpoint"):
        collect_openai_compatible(
            plan,
            configuration=collection,
            baseline_run=baseline,
            subject_run=subject,
        )
    with pytest.raises(OpenAICompatibleJudgeError, match="capacity was exhausted"):
        collect_openai_compatible(
            plan,
            configuration=collection,
            baseline_run=baseline,
            subject_run=subject,
            options=OpenAICompatibleJudgeOptions(
                checkpoint_directory=tmp_path / "capacity-checkpoint"
            ),
        )

    monkeypatch.setattr(openai_compatible, "preflight_openai_compatible", lambda *_: {})
    collection["authentication"] = "bearer_env"
    monkeypatch.delenv("INVARLOCK_OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.setattr(openai_compatible, "_NEXT_SOURCE_RESERVATION_BYTES", 1)
    with pytest.raises(OpenAICompatibleJudgeError, match="must be set"):
        collect_openai_compatible(
            plan,
            configuration=collection,
            baseline_run=baseline,
            subject_run=subject,
        )


@pytest.mark.parametrize(
    "content",
    [
        "not-json-rating",
        "x" * (1024 * 1024 + 1),
    ],
)
def test_valid_service_envelope_preserves_invalid_completion_outcome(
    tmp_path, monkeypatch, content
):
    body = _success_payload(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    )
    measurements, _, _ = _collect(tmp_path, monkeypatch, body=body)
    if len(content.encode()) > 1024 * 1024:
        assert all(
            json.loads(source["content"])["http"]["outcome"] == "malformed_response"
            for source in measurements["sources"]
        )
    else:
        assert all(
            trial["parse"]["status"] == "invalid" for trial in measurements["trials"]
        )


def test_ambiguous_admission_is_never_retried(tmp_path, monkeypatch):
    plan, collection, baseline, subject = _inputs(tmp_path)
    calls = _install_transport(monkeypatch, error=KeyboardInterrupt())
    options = OpenAICompatibleJudgeOptions(checkpoint_directory=tmp_path / "checkpoint")
    with pytest.raises(KeyboardInterrupt):
        collect_openai_compatible(
            plan,
            configuration=collection,
            baseline_run=baseline,
            subject_run=subject,
            options=options,
        )
    assert len(calls) == 1
    with pytest.raises(OpenAICompatibleJudgeError, match="cannot be retried"):
        collect_openai_compatible(
            plan,
            configuration=collection,
            baseline_run=baseline,
            subject_run=subject,
            options=options,
        )
    assert len(calls) == 1


def test_checkpoint_rejects_symlink_and_public_directory(tmp_path, monkeypatch):
    plan, collection, baseline, subject = _inputs(tmp_path)
    _install_transport(monkeypatch)
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    for checkpoint in (linked, tmp_path / "public"):
        if checkpoint.name == "public":
            checkpoint.mkdir(mode=0o755)
            checkpoint.chmod(0o755)
        with pytest.raises(OpenAICompatibleJudgeError, match="private|unsafe"):
            collect_openai_compatible(
                plan,
                configuration=collection,
                baseline_run=baseline,
                subject_run=subject,
                options=OpenAICompatibleJudgeOptions(checkpoint_directory=checkpoint),
            )


def test_active_event_loop_rejects_before_checkpoint(tmp_path):
    plan, collection, baseline, subject = _inputs(tmp_path)
    checkpoint = tmp_path / "checkpoint"

    async def invoke():
        with pytest.raises(OpenAICompatibleJudgeError, match="active event loop"):
            collect_openai_compatible(
                plan,
                configuration=collection,
                baseline_run=baseline,
                subject_run=subject,
                options=OpenAICompatibleJudgeOptions(checkpoint_directory=checkpoint),
            )

    asyncio.run(invoke())
    assert not checkpoint.exists()


def test_service_identity_is_required_only_for_endpoint_plans(tmp_path):
    plan, *_ = _inputs(tmp_path)
    missing = deepcopy(plan)
    missing["judge"].pop("service_identity")
    with pytest.raises(
        JudgeMeasurementContractError, match="require a service identity"
    ):
        validate_measurement_plan(missing)
    extra = deepcopy(plan)
    extra["judge"]["provider"] = "inspect"
    with pytest.raises(JudgeMeasurementContractError, match="only valid"):
        validate_measurement_plan(extra)


def test_whole_call_timeout_cancels_a_hanging_stream():
    class Stream:
        async def __aenter__(self):
            async def chunks(chunk_size: int):
                await asyncio.sleep(10)
                yield b"{}"

            return SimpleNamespace(status_code=200, headers={}, aiter_raw=chunks)

        async def __aexit__(self, *_: object) -> bool:
            return False

    client = SimpleNamespace(stream=lambda *_args, **_kwargs: Stream())
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(
            _bounded_exchange(
                client,
                body=b"{}",
                timeout_seconds=0.02,
                credential=None,
            )
        )
    assert time.monotonic() - started < 0.5


class _LoopbackHandler(BaseHTTPRequestHandler):
    mode = "valid"

    def log_message(self, *_: object) -> None:
        return None

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        if self.mode == "redirect":
            self.send_response(307)
            self.send_header("Location", "/v1/elsewhere")
            self.end_headers()
            return
        if self.mode == "hang":
            self.send_response(200)
            self.end_headers()
            self.wfile.flush()
            time.sleep(0.2)
            return
        body = _success_payload()
        if self.mode == "compressed":
            body = gzip.compress(body)
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.mark.parametrize(
    ("mode", "timeout", "outcome"),
    [
        ("valid", 1.0, "success"),
        ("redirect", 1.0, "http_error"),
        ("compressed", 1.0, "malformed_response"),
        ("hang", 0.02, "timeout"),
    ],
)
def test_real_httpx_loopback_transport_boundaries(mode, timeout, outcome):
    handler = type("Handler", (_LoopbackHandler,), {"mode": mode})
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except PermissionError:
        pytest.skip("the current sandbox forbids loopback socket binding")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = configuration()
    config["base_url"] = f"http://127.0.0.1:{server.server_port}/v1/"
    headers = {"accept-encoding": "identity", "content-type": "application/json"}
    try:
        with temporarily_allow_network():
            if outcome == "timeout":
                with pytest.raises(TimeoutError):
                    asyncio.run(
                        _exchange_once(
                            _client(config, headers),
                            body=b"{}",
                            credential=None,
                            timeout_seconds=timeout,
                        )
                    )
            else:
                result = asyncio.run(
                    _exchange_once(
                        _client(config, headers),
                        body=b"{}",
                        credential=None,
                        timeout_seconds=timeout,
                    )
                )
                assert result["outcome"] == outcome
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
