"""Provider failures retain closed diagnostics without credentials or error text."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from invarlock.judge_measurements import runner
from invarlock.judge_measurements.contracts import canonical_payload
from tests.judge_measurements.test_runner_boundaries import inputs as inputs
from tests.judge_measurements.test_runner_boundaries import sdk_event as sdk_event

SECRET = "Bearer secret-token https://private.invalid/path?key=secret-token"


def failure(name, module="openai", *, cause=None, status=None, code=None):
    kind = type(name, (Exception,), {"__module__": module})
    result = kind(SECRET)
    result.__cause__ = cause
    result.status_code = status
    result.code = code
    result.body = {"message": SECRET, "headers": {"Authorization": SECRET}}
    result.request_id = SECRET
    return result


@pytest.mark.parametrize(
    "error,expected,diagnostic",
    [
        (TimeoutError(SECRET), "timeout_ambiguous", "TimeoutError"),
        (failure("APIConnectionError"), "timeout_ambiguous", "APIConnectionError"),
        (
            failure(
                "APIConnectionError",
                cause=failure(
                    "ConnectError",
                    "httpx",
                    cause=failure("SSLCertVerificationError", "ssl"),
                ),
            ),
            "transport_error",
            "SSLCertVerificationError",
        ),
        (
            failure("APITimeoutError", cause=failure("ConnectTimeout", "httpx")),
            "transport_error",
            "ConnectTimeout",
        ),
        (
            failure("APIConnectionError", cause=failure("PoolTimeout", "httpcore")),
            "transport_error",
            "PoolTimeout",
        ),
        (
            failure("APIConnectionError", cause=failure("ReadError", "httpx")),
            "timeout_ambiguous",
            "ReadError",
        ),
        (
            failure("APIConnectionError", cause=failure("WriteTimeout", "httpx")),
            "timeout_ambiguous",
            "WriteTimeout",
        ),
        (
            failure(
                "APIConnectionError",
                cause=failure(
                    "ConnectError", "httpx", cause=failure("ReadTimeout", "httpx")
                ),
            ),
            "timeout_ambiguous",
            "ReadTimeout",
        ),
        (
            failure("AuthenticationError", status=401, code="invalid_api_key"),
            "timeout_ambiguous",
            "http_status=401; provider_code=invalid_api_key",
        ),
        (
            failure("RateLimitError", status=429, code="insufficient_quota"),
            "timeout_ambiguous",
            "http_status=429; provider_code=insufficient_quota",
        ),
        (
            failure("BadRequestError", status=400, code="unsupported_parameter"),
            "timeout_ambiguous",
            "http_status=400; provider_code=unsupported_parameter",
        ),
        (
            failure(
                "APIConnectionError", status=500, cause=failure("ConnectError", "httpx")
            ),
            "timeout_ambiguous",
            "http_status=500",
        ),
        (
            failure(
                "secret-token", module="untrusted", status=401, code="invalid_api_key"
            ),
            "timeout_ambiguous",
            "UnknownError",
        ),
    ],
)
def test_safe_diagnostics_distinguish_known_connection_failures_from_ambiguous_calls(
    error, expected, diagnostic
):
    status, detail = runner._safe_failure(error)
    assert status == expected
    assert diagnostic in detail["message"]
    assert set(detail) == {"code", "message"}
    assert len(detail["message"]) < 1024
    assert SECRET not in detail["message"]
    assert b"secret-token" not in canonical_payload(detail)


@pytest.mark.parametrize(
    "status,code",
    [(True, "secret-token"), (999, {"message": SECRET}), ("401", SECRET), (None, None)],
)
def test_unapproved_scalar_error_fields_are_not_serialized(status, code):
    _, detail = runner._safe_failure(
        failure("AuthenticationError", status=status, code=code)
    )
    assert "http_status=unavailable; provider_code=unavailable" in detail["message"]
    assert "secret-token" not in detail["message"]


def test_cause_chains_are_bounded_cycle_safe_and_handle_implicit_context():
    first = failure("APIConnectionError")
    second = failure("ConnectError", "httpx")
    first.__context__ = second
    second.__cause__ = first
    status, detail = runner._safe_failure(first)
    assert status == "timeout_ambiguous"
    assert detail["message"].count("APIConnectionError") == 1
    for _ in range(20):
        first = failure("APIConnectionError", cause=first)
    assert runner._safe_failure(first)[1]["message"].count("APIConnectionError") == 8


def test_suppressed_prior_connection_failure_cannot_reclassify_timeout():
    try:
        try:
            raise failure("ConnectError", "httpx")
        except Exception:
            raise TimeoutError(SECRET) from None
    except TimeoutError as error:
        status, detail = runner._safe_failure(error)
    assert status == "timeout_ambiguous"
    assert "ConnectError" not in detail["message"]


def test_truncated_chain_cannot_hide_ambiguous_read_failure():
    error = failure("ReadTimeout", "httpx")
    for _ in range(8):
        error = failure("APIConnectionError", cause=error)
    error = failure("ConnectError", "httpx", cause=error)
    status, detail = runner._safe_failure(error)
    assert status == "timeout_ambiguous"
    assert "ReadTimeout" not in detail["message"]


def test_suppressed_unknown_cause_is_not_proof_of_an_unsent_call():
    error = failure("ConnectError", "httpx")
    error.__context__ = RuntimeError(SECRET)
    error.__suppress_context__ = True
    assert runner._safe_failure(error)[0] == "timeout_ambiguous"


def test_implicit_prior_connection_failure_is_diagnostic_not_dispatch_proof():
    error = TimeoutError(SECRET)
    error.__context__ = failure("ConnectError", "httpx")
    status, detail = runner._safe_failure(error)
    assert status == "timeout_ambiguous"
    assert "TimeoutError>ConnectError" in detail["message"]


def test_unknown_wrapper_cannot_establish_an_unsent_call():
    error = failure(
        "PrivateWrapper", "unrecognized", cause=failure("ConnectError", "httpx")
    )
    assert runner._safe_failure(error)[0] == "timeout_ambiguous"


@pytest.mark.parametrize(
    "module", ["httpx2", "httpx2._exceptions", "httpcore2", "httpcore2._exceptions"]
)
@pytest.mark.parametrize("transport", ["ConnectError", "ReadError"])
def test_current_sdk_transport_namespaces_keep_exact_known_exception_names(
    module, transport
):
    error = failure(
        "ModelGenerateError",
        "inspect_ai.model._model",
        cause=failure("APIConnectionError", cause=failure(transport, module)),
    )
    status, detail = runner._safe_failure(error)
    assert status == (
        "transport_error" if transport == "ConnectError" else "timeout_ambiguous"
    )
    assert "ModelGenerateError>APIConnectionError>" + transport in detail["message"]
    assert b"secret-token" not in canonical_payload(detail)


@pytest.mark.parametrize("event_error", [None, SECRET])
def test_upstream_event_error_takes_precedence_over_output_refusal_marker(
    inputs, sdk_event, event_error
):
    sdk_event.error = event_error
    sdk_event.output.error = SECRET
    sdk_event.output.metadata = {"request_id": SECRET}
    projected = runner._project_event(
        sdk_event,
        expected_request=inputs["export"]["samples"][0]["events"][0]["call"]["request"],
        options=inputs["options"],
        failure_status=None,
    )
    assert projected["error"]["status"] == (
        "timeout_ambiguous" if event_error is not None else "refusal"
    )
    assert b"secret-token" not in canonical_payload(projected)


@pytest.mark.parametrize("phase", ["pending", "complete", "absent"])
@pytest.mark.parametrize("kind", ["tls", "http", "timeout", "unknown"])
def test_call_one_preserves_safe_cause_even_when_inspect_completes_error_event(
    inputs, sdk_event, monkeypatch, phase, kind
):
    exceptions = {
        "tls": failure(
            "APIConnectionError",
            cause=failure(
                "ConnectError",
                "httpx",
                cause=failure("SSLCertVerificationError", "ssl"),
            ),
        ),
        "http": failure("AuthenticationError", status=401, code="invalid_api_key"),
        "timeout": TimeoutError(SECRET),
        "unknown": RuntimeError(SECRET),
    }
    exception = exceptions[kind]
    request = inputs["export"]["samples"][0]["events"][0]["call"]["request"]
    state = {}

    @contextmanager
    def sink(value):
        state["sink"] = value
        yield

    class Model:
        async def generate(self, **kwargs):
            if phase == "absent":
                raise exception
            sdk_event.error = SECRET
            # A provider error must not be mistaken for a model-content refusal.
            sdk_event.output.error = SECRET
            sdk_event.output.metadata = {"request_id": SECRET}
            sdk_event.call.response = {"error": {"code": SECRET, "message": SECRET}}
            state["sink"].on_pending(sdk_event)
            if phase == "complete":
                state["sink"].on_complete(sdk_event)
            raise exception

    model_module = SimpleNamespace(
        **{
            name: lambda **values: SimpleNamespace(**values)
            for name in ("ChatMessageSystem", "ChatMessageUser", "ChatMessageAssistant")
        }
    )
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(use_model_event_sink=sink)
            if name.endswith("._model")
            else model_module
        ),
    )

    def call():
        return asyncio.run(
            runner._call_one(
                Model(),
                request=request,
                config=None,
                options=inputs["options"],
                pacer=runner._Pacer(0),
                check_directory=lambda: None,
            )
        )

    if phase == "absent":
        with pytest.raises(
            runner.InspectJudgeError, match="exception_chain"
        ) as observed:
            call()
        assert "secret-token" not in str(observed.value)
        return
    event = call()
    assert event["error"]["status"] == (
        "transport_error" if kind == "tls" else "timeout_ambiguous"
    )
    assert event["error"]["status"] != "refusal"
    assert ("http_status=401" in event["error"]["message"]) == (kind == "http")
    assert b"secret-token" not in canonical_payload(event)
    assert event["call"]["response"] is None
    assert event["output"]["request_id"] is None
    assert event["output"]["completion"] == ""
