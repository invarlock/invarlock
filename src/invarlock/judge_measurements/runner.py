"""Bounded live collection through one caller-supplied Inspect model."""

from __future__ import annotations

import asyncio
import fcntl
import importlib
import importlib.metadata
import os
import stat
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

from invarlock.captured_contracts import _read_at
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.filesystem.atomic_file import write_file_no_replace
from invarlock.filesystem.paths import (
    PathChangedError,
    entry_identity,
    pinned_directory,
)
from invarlock.judge_measurement_types import JudgeMeasurementPlan, JudgeMeasurements
from invarlock.judge_measurements.contracts import (
    JudgeMeasurementContractError,
    _check_inspect_provider_projection,
    _check_inspect_provider_response,
    canonical_payload,
    expected_trial_id,
    measurement_plan_digest,
)

from .collector import (
    EXPORT_FORMAT,
    EXPORT_PROFILE,
    INSPECT_VERSION,
    MAX_RETAINED_EVENT_BYTES,
    CollectionOptions,
    InspectJudgeError,
    _check_options,
    _LiveCheckpoint,
    _render_request,
    import_export,
    prepare_inspect_config,
)

_HEADER = "collection.json"
_ADMISSION_PREFIX = "admission-"
_RESULT_PREFIX = "result-"
_LOCK = ".collection.lock"
_PROVIDER_ERROR_TYPES = frozenset(
    {
        "APIError",
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "APIResponseValidationError",
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "ConflictError",
        "UnprocessableEntityError",
        "RateLimitError",
        "InternalServerError",
    }
)
_TRANSPORT_ERROR_TYPES = frozenset(
    {
        "ConnectError",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "WriteError",
        "WriteTimeout",
        "RemoteProtocolError",
        "LocalProtocolError",
        "ProxyError",
        "TimeoutException",
    }
)
_ERROR_TYPES = {
    "openai": _PROVIDER_ERROR_TYPES,
    "openai._exceptions": _PROVIDER_ERROR_TYPES,
    "anthropic": _PROVIDER_ERROR_TYPES,
    "anthropic._exceptions": _PROVIDER_ERROR_TYPES,
    "httpx": _TRANSPORT_ERROR_TYPES,
    "httpx._exceptions": _TRANSPORT_ERROR_TYPES,
    "httpx2": _TRANSPORT_ERROR_TYPES,
    "httpx2._exceptions": _TRANSPORT_ERROR_TYPES,
    "httpcore": _TRANSPORT_ERROR_TYPES,
    "httpcore._exceptions": _TRANSPORT_ERROR_TYPES,
    "httpcore2": _TRANSPORT_ERROR_TYPES,
    "httpcore2._exceptions": _TRANSPORT_ERROR_TYPES,
    "inspect_ai.model._model": frozenset({"ModelGenerateError", "AttemptTimeoutError"}),
    "builtins": frozenset(
        {
            "TimeoutError",
            "ConnectionRefusedError",
            "ConnectionResetError",
            "ConnectionAbortedError",
            "ValueError",
            "TypeError",
            "RuntimeError",
            "OSError",
            "PermissionError",
            "FileNotFoundError",
        }
    ),
    "ssl": frozenset({"SSLError", "SSLCertVerificationError"}),
    "socket": frozenset({"gaierror"}),
}
_PROVIDER_ERROR_CODES = frozenset(
    {
        "invalid_api_key",
        "insufficient_quota",
        "rate_limit_exceeded",
        "context_length_exceeded",
        "model_not_found",
        "invalid_request_error",
        "unsupported_value",
        "invalid_value",
        "unsupported_parameter",
        "content_policy_violation",
        "server_error",
        "account_deactivated",
    }
)


def _safe_failure(exception: Exception) -> tuple[str, dict[str, str]]:
    """Retain only closed diagnostic identifiers, never provider error text."""
    names: list[str] = []
    seen: set[int] = set()
    status_code: int | None = None
    provider_code: str | None = None
    incomplete = False
    current: BaseException | None = exception
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        kind = type(current)
        known = kind.__name__ in _ERROR_TYPES.get(kind.__module__, ())
        names.append(kind.__name__ if known else "UnknownError")
        if known and kind.__module__.split(".", 1)[0] in {"openai", "anthropic"}:
            fields = vars(current)
            status = fields.get("status_code")
            if status_code is None and type(status) is int and 100 <= status <= 599:
                status_code = status
            code = fields.get("code")
            if (
                provider_code is None
                and type(code) is str
                and code in _PROVIDER_ERROR_CODES
            ):
                provider_code = code
        cause = current.__cause__
        if cause is None:
            if current.__suppress_context__:
                incomplete = incomplete or current.__context__ is not None
            else:
                cause = current.__context__
                # An exception raised while handling another is not proof that
                # the earlier operation caused the current failure.
                incomplete = incomplete or cause is not None
        current = cause
    incomplete = incomplete or current is not None
    # Connection/pool failures precede sending application bytes. A provider's
    # generic APIConnectionError can wrap read/write failures, so is insufficient.
    before_send = bool({"ConnectError", "ConnectTimeout", "PoolTimeout"} & set(names))
    ambiguous = bool(
        {
            "ReadError",
            "ReadTimeout",
            "WriteError",
            "WriteTimeout",
            "RemoteProtocolError",
            "LocalProtocolError",
        }
        & set(names)
    )
    transport = (
        before_send
        and not ambiguous
        and not incomplete
        and "UnknownError" not in names
        and status_code is None
    )
    return (
        "transport_error" if transport else "timeout_ambiguous",
        {
            "code": "inspect-connect-failed" if transport else "inspect-call-failed",
            "message": (
                "Inspect judge call failed; exception_chain="
                + ">".join(names)
                + f"; http_status={status_code if status_code is not None else 'unavailable'}"
                + f"; provider_code={provider_code or 'unavailable'}"
            ),
        },
    )


@dataclass(frozen=True)
class RunnerOptions:
    """Execution limits and scorer identity for a durable checkpoint.

    ``stop_after_batches`` stops this invocation after fully retained batches.
    It is excluded from checkpoint identity, like the invocation timeout, so a
    later invocation can resume with the same plan and collection budgets.
    """

    checkpoint_directory: Path
    scorer_id: str
    invocation_timeout_seconds: int
    stop_after_batches: int | None = None

    def validate(self) -> None:
        if not isinstance(self.checkpoint_directory, Path):
            raise InspectJudgeError("checkpoint_directory must be a Path")
        if (
            not isinstance(self.scorer_id, str)
            or not self.scorer_id
            or len(self.scorer_id) > 128
            or any(
                char
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for char in self.scorer_id
            )
        ):
            raise InspectJudgeError("scorer_id must be a bounded identifier")
        if (
            type(self.invocation_timeout_seconds) is not int
            or not 1 <= self.invocation_timeout_seconds <= 7 * 24 * 60 * 60
        ):
            raise InspectJudgeError(
                "invocation_timeout_seconds must be between 1 and 604800"
            )
        if self.stop_after_batches is not None and (
            type(self.stop_after_batches) is not int
            or not 1 <= self.stop_after_batches <= 1_000_000
        ):
            raise InspectJudgeError("stop_after_batches must be between 1 and 1000000")


class _EventSink:
    def __init__(self) -> None:
        self.pending: Any | None = None
        self.complete: Any | None = None

    def on_pending(self, event: Any) -> None:
        if self.pending is not None:
            raise InspectJudgeError("one judge call emitted more than one model event")
        self.pending = event

    def on_complete(self, event: Any) -> None:
        if self.complete is not None:
            raise InspectJudgeError(
                "one judge call completed more than one model event"
            )
        self.complete = event


class _Pacer:
    def __init__(self, spacing: float) -> None:
        self._spacing = spacing
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            delay = self._last + self._spacing - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


def _header(
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
) -> dict[str, Any]:
    return {
        "format": EXPORT_FORMAT,
        "profile": EXPORT_PROFILE,
        "inspect_version": INSPECT_VERSION,
        "plan_sha256": measurement_plan_digest(plan),
        "collection": asdict(options),
        "scorer_id": runner.scorer_id,
    }


def _initialize_checkpoint(
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
    directory_fd: int,
) -> None:
    runner.validate()
    directory = runner.checkpoint_directory
    expected = canonical_payload(_header(plan, options, runner))
    path = directory / _HEADER
    if _HEADER in os.listdir(directory_fd):
        observed, _ = _read_at(directory_fd, _HEADER, 65536)
        if observed != expected:
            raise InspectJudgeError("checkpoint identity differs from this collection")
    else:
        write_file_no_replace(path, expected)


def _empty_export(
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
) -> dict[str, Any]:
    digest = measurement_plan_digest(plan)
    samples = []
    for binding in sorted(plan["answer_bindings"], key=lambda row: row["case_id"]):
        binding_raw = cast(dict[str, Any], binding)
        for side in ("baseline", "subject"):
            for repetition in range(1, plan["schedule"]["repetitions"] + 1):
                samples.append(
                    {
                        "id": expected_trial_id(
                            digest, binding["case_id"], side, repetition
                        ),
                        "epoch": 1,
                        "metadata": {
                            "case_id": binding["case_id"],
                            "side": side,
                            "repetition": repetition,
                            "plan_sha256": digest,
                            "answer_sha256": binding_raw[f"{side}_answer_sha256"],
                            "scorer_id": runner.scorer_id,
                        },
                        "events": [],
                    }
                )
    return {
        "format": EXPORT_FORMAT,
        "profile": EXPORT_PROFILE,
        "inspect_version": INSPECT_VERSION,
        "collection": asdict(options),
        "samples": samples,
    }


def _checkpoint_export(
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
    directory_fd: int,
) -> dict[str, Any]:
    exported = _empty_export(plan, options, runner)
    by_id = {sample["id"]: sample for sample in exported["samples"]}
    records: dict[tuple[str, int], dict[str, Any]] = {}
    admissions: set[tuple[str, int]] = set()
    results: set[tuple[str, int]] = set()
    for name in sorted(os.listdir(directory_fd)):
        path = Path(name)
        if path.name in {_HEADER, _LOCK}:
            continue
        is_admission = path.name.startswith(_ADMISSION_PREFIX)
        is_result = path.name.startswith(_RESULT_PREFIX)
        if not (is_admission or is_result) or path.suffix != ".json":
            raise InspectJudgeError("checkpoint directory contains an unknown entry")
        payload, _ = _read_at(directory_fd, name, MAX_RETAINED_EVENT_BYTES + 4096)
        record = parse_json_bytes(payload, label="Inspect judge checkpoint attempt")
        if not isinstance(record, dict) or set(record) != {
            "trial_id",
            "attempt",
            "event",
        }:
            raise InspectJudgeError("checkpoint attempt has an unsupported shape")
        trial_id = record["trial_id"]
        attempt = record["attempt"]
        if trial_id not in by_id or type(attempt) is not int:
            raise InspectJudgeError("checkpoint attempt has an unknown identity")
        key = (trial_id, attempt)
        if is_admission:
            if key in admissions:
                raise InspectJudgeError("checkpoint contains duplicate admissions")
            admissions.add(key)
            records.setdefault(key, record)
        else:
            if key in results:
                raise InspectJudgeError("checkpoint contains duplicate results")
            results.add(key)
            records[key] = record
    if any(key not in admissions for key in results):
        raise InspectJudgeError("checkpoint result has no prior call admission")
    for (trial_id, attempt), record in sorted(records.items()):
        events = by_id[trial_id]["events"]
        if attempt != len(events) + 1:
            raise InspectJudgeError(
                "checkpoint attempts are missing, duplicated, or reordered"
            )
        events.append(record["event"])
    return exported


def _checkpoint_path(
    runner: RunnerOptions, prefix: str, trial_id: str, attempt: int
) -> Path:
    return runner.checkpoint_directory / f"{prefix}{trial_id}-{attempt:02d}.json"


@contextmanager
def _collection_lock(directory_fd: int) -> Iterator[int]:
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(_LOCK, flags, 0o600, dir_fd=directory_fd)
    try:
        current = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_uid != os.geteuid()
            or stat.S_IMODE(current.st_mode) & 0o077
        ):
            raise InspectJudgeError("collection lock is not a safe regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InspectJudgeError(
                "another collector is already using this checkpoint"
            ) from exc
        try:
            yield descriptor
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _admission_event(
    *,
    trial_id: str,
    attempt: int,
    request: dict[str, Any],
    options: CollectionOptions,
) -> dict[str, Any]:
    config = request["config"]
    return {
        "event": "model",
        "uuid": f"admission-{trial_id}-{attempt}",
        "role": "grader",
        "model": options.grader,
        "input": request["messages"],
        "tools": [],
        "tool_choice": "none",
        "config": {
            "temperature": float(config["temperature"]),
            "top_p": float(config["top_p"]),
            "max_tokens": config["max_output_tokens"],
            "seed": config["seed"],
            "reasoning_effort": config["reasoning_effort"],
            "max_retries": 0,
            "timeout": options.request_timeout_seconds,
            "attempt_timeout": options.request_timeout_seconds,
            "max_connections": options.concurrency,
            "adaptive_connections": False,
            "num_choices": 1,
            "internal_tools": False,
            "parallel_tool_calls": False,
            "reasoning_summary": "none",
            "reasoning_history": "none",
            "cache": False,
            "batch": False,
        },
        "retries": 0,
        "cache": None,
        "call": {"request": request, "response": None, "error": True},
        "output": {
            "model": None,
            "request_id": None,
            "finish_reason": None,
            "usage": None,
            "completion": "",
        },
        "error": {
            "status": "timeout_ambiguous",
            "code": "call-admitted",
            "message": "Call was admitted but no completed checkpoint was retained.",
        },
    }


def _require_provider_retries_disabled(model: Any) -> None:
    api = getattr(model, "api", None)
    if (
        str(model).startswith("anthropic/")
        and getattr(api, "_invarlock_single_attempt", False) is not True
    ):
        raise InspectJudgeError(
            "Anthropic provider requires the configured single-attempt client"
        )
    client = getattr(api, "client", None)
    if client is None and str(model).startswith("google/"):
        if getattr(api, "_invarlock_single_attempt", False) is True:
            return
        raise InspectJudgeError(
            "Google provider requires the configured single-attempt client"
        )
    if getattr(client, "max_retries", None) != 0:
        raise InspectJudgeError("the Inspect provider client must expose max_retries=0")


def _require_clean_model_configuration(model: Any) -> None:
    model_config = getattr(model, "config", None)
    dump = getattr(model_config, "model_dump", None)
    if dump is None:
        raise InspectJudgeError("Inspect model configuration is unavailable")
    if dump(exclude_none=True):
        raise InspectJudgeError(
            "Inspect model configuration must not contain inherited settings"
        )
    api = getattr(model, "api", None)
    responses_api = getattr(api, "responses_api", None)
    if responses_api not in (None, False):
        raise InspectJudgeError(
            "Inspect provider must explicitly use the chat-completion API"
        )
    if getattr(api, "service_tier", None) not in (None, "default"):
        raise InspectJudgeError("Inspect provider contains an unsupported service tier")
    allowed_model_args = {
        "max_retries": 0,
        "responses_api": False,
        "service_tier": "default",
    }
    for label, model_args in (
        ("model", getattr(model, "model_args", {})),
        ("provider", getattr(api, "model_args", {})),
    ):
        if not isinstance(model_args, dict) or any(
            key not in allowed_model_args or allowed_model_args[key] != value
            for key, value in model_args.items()
        ):
            raise InspectJudgeError(
                f"Inspect {label} model_args contain unsupported settings"
            )


def _request_id(response: Any, output: Any) -> str | None:
    for value in (
        response.get("id") if isinstance(response, dict) else None,
        response.get("request_id") if isinstance(response, dict) else None,
        response.get("response_id") if isinstance(response, dict) else None,
        response.get("responseId") if isinstance(response, dict) else None,
        getattr(output, "metadata", None),
    ):
        if isinstance(value, str) and 0 < len(value) <= 256:
            return value
        if isinstance(value, dict):
            candidate = value.get("request_id")
            if isinstance(candidate, str) and 0 < len(candidate) <= 256:
                return candidate
    return None


def _check_openai_wire_call(
    request: dict[str, Any], response: Any, expected_request: dict[str, Any]
) -> None:
    try:
        _check_inspect_provider_projection(
            call={"request": request, "response": response},
            output={},
            normalized_request=expected_request,
            completed=False,
            inspect_version=INSPECT_VERSION,
        )
    except JudgeMeasurementContractError as exc:
        raise InspectJudgeError(str(exc)) from None
    if isinstance(response, dict) and response.get("service_tier") not in (
        None,
        "default",
    ):
        raise InspectJudgeError(
            "provider response contains an unsupported service tier"
        )


def _wire_tokens(usage: dict[str, Any], name: str, *, optional: bool = False) -> int:
    value = usage.get(name)
    if value is None and optional:
        return 0
    if type(value) is not int or value < 0:
        raise InspectJudgeError(
            "provider response token usage must use nonnegative integers"
        )
    return value


def _anthropic_wire_response(response: dict[str, Any]) -> dict[str, Any]:
    content = response.get("content")
    usage = response.get("usage")
    if (
        response.get("role") != "assistant"
        or not isinstance(content, list)
        or not isinstance(usage, dict)
    ):
        raise InspectJudgeError("Anthropic provider response has an unsupported shape")
    if usage.get("iterations"):
        raise InspectJudgeError(
            "Anthropic provider response contains unsupported iterations"
        )
    texts = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in {
            "text",
            "thinking",
            "redacted_thinking",
        }:
            raise InspectJudgeError(
                "Anthropic provider response contains unsupported content"
            )
        if part["type"] == "text":
            if not isinstance(part.get("text"), str):
                raise InspectJudgeError("Anthropic provider response text is invalid")
            texts.append(part["text"])
    reasons = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "max_tokens",
        "refusal": "content_filter",
    }
    if response.get("stop_reason") not in reasons:
        raise InspectJudgeError(
            "Anthropic provider response finish reason is unsupported"
        )
    return {
        "format": "invarlock/judge-provider-response-v1",
        "content": "\n".join(texts),
        "model": response.get("model"),
        "id": response.get("id"),
        "finish_reason": reasons[response["stop_reason"]],
        "usage": {
            "input_tokens": _wire_tokens(usage, "input_tokens")
            + _wire_tokens(usage, "cache_read_input_tokens", optional=True)
            + _wire_tokens(usage, "cache_creation_input_tokens", optional=True),
            "output_tokens": _wire_tokens(usage, "output_tokens"),
        },
    }


def _google_wire_response(response: dict[str, Any]) -> dict[str, Any]:
    candidates = response.get("candidates")
    usage = response.get("usageMetadata")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 1
        or not isinstance(candidates[0], dict)
        or not isinstance(usage, dict)
    ):
        raise InspectJudgeError("Google provider response has an unsupported shape")
    candidate = candidates[0]
    content = candidate.get("content") or {}
    if not isinstance(content, dict) or content.get("role") not in (None, "model"):
        raise InspectJudgeError("Google provider response content is invalid")
    parts = content.get("parts") or []
    if not isinstance(parts, list):
        raise InspectJudgeError("Google provider response content is invalid")
    texts = []
    for part in parts:
        if not isinstance(part, dict) or any(
            value is not None
            for key, value in part.items()
            if key not in {"text", "thought", "thoughtSignature"}
        ):
            raise InspectJudgeError(
                "Google provider response contains unsupported content"
            )
        if part.get("text") is not None:
            if not isinstance(part["text"], str):
                raise InspectJudgeError("Google provider response text is invalid")
            if part.get("thought") is not True:
                texts.append(part["text"])
    reasons = {
        "STOP": "stop",
        "MAX_TOKENS": "max_tokens",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "BLOCKLIST": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "SPII": "content_filter",
    }
    reason = candidate.get("finishReason") or "STOP"
    if reason not in reasons:
        raise InspectJudgeError("Google provider response finish reason is unsupported")
    return {
        "format": "invarlock/judge-provider-response-v1",
        "content": "\n".join(texts),
        "model": response.get("modelVersion"),
        "id": response.get("responseId"),
        "finish_reason": reasons[reason],
        "usage": {
            "input_tokens": _wire_tokens(usage, "promptTokenCount"),
            "output_tokens": _wire_tokens(usage, "candidatesTokenCount", optional=True)
            + _wire_tokens(usage, "thoughtsTokenCount", optional=True),
        },
    }


def _check_wire_response(
    response: dict[str, Any],
    output: dict[str, Any],
    request: dict[str, Any],
    grader: str,
) -> None:
    provider = grader.partition("/")[0]
    if provider == "anthropic":
        response = _anthropic_wire_response(response)
        request = {}
    elif provider == "google":
        response = _google_wire_response(response)
        request = {}
    elif provider in {"openai", "openrouter"}:
        if not isinstance(response.get("choices"), list) or not isinstance(
            response.get("usage"), dict
        ):
            raise InspectJudgeError(
                "provider response is missing its completion or usage"
            )
        _wire_tokens(response["usage"], "prompt_tokens")
        _wire_tokens(response["usage"], "completion_tokens")
    try:
        _check_inspect_provider_response(
            response=response, output=output, request=request
        )
    except JudgeMeasurementContractError as exc:
        raise InspectJudgeError(str(exc)) from None


def _project_event(
    event: Any,
    *,
    expected_request: dict[str, Any],
    options: CollectionOptions,
    failure_status: str | None,
    failure_error: dict[str, str] | None = None,
) -> dict[str, Any]:
    def public_message(message: Any) -> dict[str, str]:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise InspectJudgeError("Inspect model input must use plain text messages")
        return {"role": role, "content": content}

    event_input = getattr(event, "input", None)
    if not isinstance(event_input, list):
        raise InspectJudgeError("Inspect model event is missing its input")
    projected_input = [public_message(message) for message in event_input]
    if projected_input != expected_request["messages"]:
        raise InspectJudgeError("Inspect model input differs from the approved request")
    event_model = getattr(event, "model", None)
    if event_model != options.grader:
        raise InspectJudgeError("Inspect model event differs from the approved grader")
    if getattr(event, "tools", None) != []:
        raise InspectJudgeError("Inspect model event used tools")
    tool_choice = getattr(event, "tool_choice", None)
    if tool_choice != "none":
        raise InspectJudgeError("Inspect model event used an unsupported tool choice")
    retries = getattr(event, "retries", 0)
    if retries not in (None, 0):
        raise InspectJudgeError("Inspect model event contains SDK retries")
    if getattr(event, "cache", None) is not None:
        raise InspectJudgeError("Inspect model event used a cache")
    output = getattr(event, "output", None)
    call = getattr(event, "call", None)
    call_request = getattr(call, "request", None)
    call_response = getattr(call, "response", None)
    call_error = getattr(call, "error", None)
    if not isinstance(call_request, dict):
        raise InspectJudgeError("Inspect did not retain the provider request")
    if call_response is not None and not isinstance(call_response, dict):
        raise InspectJudgeError("Inspect provider response must be a JSON object")
    if (
        options.grader.startswith("openai/")
        and failure_status is None
        and getattr(event, "error", None) is None
    ):
        _check_openai_wire_call(call_request, call_response, expected_request)
    usage = getattr(output, "usage", None)
    projected_usage = None
    if usage is not None:

        def tokens(name: str, *, optional: bool = False) -> int:
            value = getattr(usage, name, None)
            if value is None and optional:
                return 0
            if type(value) is not int or value < 0:
                raise InspectJudgeError(
                    "provider token usage must use nonnegative integers"
                )
            return value

        input_tokens = tokens("input_tokens")
        input_tokens += tokens("input_tokens_cache_read", optional=True)
        input_tokens += tokens("input_tokens_cache_write", optional=True)
        projected_usage = {
            "input_tokens": input_tokens,
            "output_tokens": tokens("output_tokens"),
        }
        total_cost = getattr(usage, "total_cost", None)
        if total_cost is not None:
            try:
                cost = Decimal(str(total_cost)) * 1_000_000
            except InvalidOperation:
                raise InspectJudgeError(
                    "provider cost must be a finite nonnegative amount"
                ) from None
            if not cost.is_finite() or cost < 0:
                raise InspectJudgeError(
                    "provider cost must be a finite nonnegative amount"
                )
            if cost > options.cost_microusd_per_call:
                raise InspectJudgeError(
                    "provider cost exceeded the reserved per-call amount"
                )
    resolved_model = getattr(output, "model", None)
    completion = getattr(output, "completion", "")
    stop_reason = None
    choices = getattr(output, "choices", None)
    if choices:
        if len(choices) != 1:
            raise InspectJudgeError("Inspect model output must contain one completion")
        message = getattr(choices[0], "message", None)
        if getattr(message, "tool_calls", None):
            raise InspectJudgeError("Inspect model output contains tool calls")
        stop_reason = getattr(choices[0], "stop_reason", None)
    event_error = getattr(event, "error", None)
    if failure_status is None and (
        event_error is not None or getattr(output, "error", None)
    ):
        failure_status = "timeout_ambiguous" if event_error is not None else "refusal"
    error = None
    if failure_status is not None:
        error = {
            "status": failure_status,
            **(
                failure_error
                or {
                    "code": "inspect-call-failed",
                    "message": "Inspect judge call did not complete",
                }
            ),
        }
        resolved_model = None
        projected_usage = None
        stop_reason = None
        call_response = None
        completion = ""
    config = expected_request["config"]
    event_config = getattr(event, "config", None)
    projected_config = {
        "temperature": getattr(event_config, "temperature", None),
        "top_p": getattr(event_config, "top_p", None),
        "max_tokens": getattr(event_config, "max_tokens", None),
        "seed": getattr(event_config, "seed", None),
        "reasoning_effort": getattr(event_config, "reasoning_effort", None),
        "max_retries": getattr(event_config, "max_retries", None),
        "timeout": getattr(event_config, "timeout", None),
        "attempt_timeout": getattr(event_config, "attempt_timeout", None),
        "max_connections": getattr(event_config, "max_connections", None),
        "adaptive_connections": getattr(event_config, "adaptive_connections", None),
        "num_choices": getattr(event_config, "num_choices", None),
        "internal_tools": getattr(event_config, "internal_tools", None),
        "parallel_tool_calls": getattr(event_config, "parallel_tool_calls", None),
        "reasoning_summary": getattr(event_config, "reasoning_summary", None),
        "reasoning_history": getattr(event_config, "reasoning_history", None),
        "cache": getattr(event_config, "cache", None),
        "batch": getattr(event_config, "batch", None),
    }
    expected_config = {
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "max_tokens": config["max_output_tokens"],
        "seed": config["seed"],
        "reasoning_effort": config["reasoning_effort"],
        "max_retries": 0,
        "timeout": options.request_timeout_seconds,
        "attempt_timeout": options.request_timeout_seconds,
        "max_connections": options.concurrency,
        "adaptive_connections": False,
        "num_choices": 1,
        "internal_tools": False,
        "parallel_tool_calls": False,
        "reasoning_summary": "none",
        "reasoning_history": "none",
        "cache": False,
        "batch": False,
    }
    if projected_config != expected_config:
        raise InspectJudgeError("Inspect model event has changed generation settings")
    dump_config = getattr(event_config, "model_dump", None)
    retained_config = {
        key: value for key, value in expected_config.items() if value is not None
    }
    if dump_config is not None and dump_config(exclude_none=True) != retained_config:
        raise InspectJudgeError(
            "Inspect model event contains hidden generation settings"
        )
    retained_response = None
    if call_response is not None:
        _check_wire_response(
            call_response,
            {
                "completion": completion,
                "model": resolved_model,
                "request_id": _request_id(call_response, output),
                "finish_reason": stop_reason,
                "usage": projected_usage,
            },
            call_request,
            options.grader,
        )
        retained_response = {
            "format": "invarlock/judge-provider-response-v1",
            "content": completion if isinstance(completion, str) else str(completion),
            "model": resolved_model,
            "id": _request_id(call_response, output),
            "finish_reason": stop_reason,
            "usage": projected_usage,
        }
    return {
        "event": "model",
        "uuid": str(getattr(event, "uuid", None) or "missing-event-id"),
        "role": "grader",
        "model": event_model,
        "input": projected_input,
        "tools": [],
        "tool_choice": "none",
        "config": projected_config,
        "retries": 0,
        "cache": getattr(event, "cache", None),
        "call": {
            # Provider SDK wire shapes differ. The authenticated Inspect input,
            # config, and output are retained in one provider-neutral envelope;
            # raw transport objects are validated above but never become the
            # long-lived judge contract.
            "request": expected_request,
            "response": retained_response,
            "error": True if error is not None else call_error,
        },
        "output": {
            "model": resolved_model,
            "request_id": _request_id(call_response, output) if error is None else None,
            "finish_reason": stop_reason,
            "usage": projected_usage,
            "completion": completion
            if isinstance(completion, str)
            else str(completion),
        },
        "error": error,
    }


def _frozen_rows(
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    *,
    per_case_reference: bool = False,
) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for side, run in (("baseline", baseline_run), ("subject", subject_run)):
        for record in run.get("records", []):
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise InspectJudgeError("frozen run contains an invalid record")
            row = rows.setdefault(record["id"], {})
            if "input" in row and row["input"] != record.get("input"):
                raise InspectJudgeError("frozen paired inputs differ")
            if not isinstance(record.get("input"), str) or not isinstance(
                record.get("output"), str
            ):
                raise InspectJudgeError("frozen runs require text inputs and outputs")
            row["input"] = record["input"]
            row[side] = record["output"]
            if per_case_reference:
                expected = record.get("expected")
                if not isinstance(expected, str):
                    raise InspectJudgeError(
                        "per-case judging requires a string reference"
                    )
                if "expected" in row and row["expected"] != expected:
                    raise InspectJudgeError("frozen paired references differ")
                row["expected"] = expected
    return rows


async def _call_one(
    model: Any,
    *,
    request: dict[str, Any],
    config: Any,
    options: CollectionOptions,
    pacer: _Pacer,
    check_directory: Callable[[], None],
) -> dict[str, Any]:
    await pacer.wait()
    check_directory()
    model_module = importlib.import_module("inspect_ai.model")
    sink_module = importlib.import_module("inspect_ai.model._model")
    sink = _EventSink()
    messages = []
    for message in request["messages"]:
        classes = {
            "system": model_module.ChatMessageSystem,
            "user": model_module.ChatMessageUser,
            "assistant": model_module.ChatMessageAssistant,
        }
        cls = classes[message["role"]]
        messages.append(cls(content=message["content"]))
    failure_status = None
    failure_error = None
    try:
        with sink_module.use_model_event_sink(sink):
            async with asyncio.timeout(options.request_timeout_seconds):
                await model.generate(
                    input=messages,
                    tools=[],
                    tool_choice="none",
                    config=config,
                    cache=False,
                )
    except TimeoutError as exc:
        failure_status, failure_error = _safe_failure(exc)
    except Exception as exc:
        if (
            sink.complete is None
            or getattr(sink.complete, "error", None) is not None
            or getattr(getattr(sink.complete, "output", None), "error", None)
        ):
            failure_status, failure_error = _safe_failure(exc)
    event = sink.complete or sink.pending
    if event is None:
        detail = f"; {failure_error['message']}" if failure_error is not None else ""
        raise InspectJudgeError(
            "Inspect did not expose a model event for the call" + detail
        ) from None
    return _project_event(
        event,
        expected_request=request,
        options=options,
        failure_status=failure_status,
        failure_error=failure_error,
    )


async def _collect_pinned(
    *,
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
    model: Any,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    directory_fd: int,
    directory_bindings: tuple[tuple[Path, tuple[int, int, int]], ...],
    on_stop: Callable[[str], None] | None = None,
) -> JudgeMeasurements:
    """Collect or resume fixed-answer judgments with durable attempt shards.

    The caller creates the Inspect model and therefore owns provider credentials.
    This function never accepts or writes credentials, provider URLs, or headers.
    """
    _check_options(plan, options)
    runner.validate()
    if plan["schedule"]["max_attempts"] != 1:
        raise InspectJudgeError(
            "live Inspect collection currently requires max_attempts=1"
        )
    if importlib.metadata.version("inspect-ai") != INSPECT_VERSION:
        raise InspectJudgeError("unsupported installed Inspect version")
    if str(model) != options.grader:
        raise InspectJudgeError(
            "Inspect model identity differs from the approved grader"
        )
    _require_provider_retries_disabled(model)
    _require_clean_model_configuration(model)

    def check_directory() -> None:
        for path, identity in directory_bindings:
            if entry_identity(path.stat(follow_symlinks=False)) != identity:
                raise PathChangedError("checkpoint directory ancestry changed")

    check_directory()
    _initialize_checkpoint(plan, options, runner, directory_fd)
    check_directory()
    with _collection_lock(directory_fd):
        config = prepare_inspect_config(plan, options)
        rows = _frozen_rows(
            baseline_run,
            subject_run,
            per_case_reference=plan["prompt"].get("reference_mode") == "per_case",
        )
        pacer = _Pacer(60 / options.requests_per_minute)
        started = time.monotonic()

        def replay() -> tuple[dict[str, Any], JudgeMeasurements]:
            check_directory()
            exported = _checkpoint_export(plan, options, runner, directory_fd)
            checkpoint = import_export(
                canonical_payload(exported),
                plan=plan,
                options=options,
                baseline_run=baseline_run,
                subject_run=subject_run,
            )
            check_directory()
            return exported, checkpoint

        def stopped(reason: str) -> JudgeMeasurements:
            result = replay()[1]
            if on_stop is not None:
                on_stop(reason)
            return result

        exported, checkpoint = replay()
        state = _LiveCheckpoint(
            plan=plan,
            options=options,
            exported=exported,
            checkpoint=checkpoint,
            frozen_inputs=rows,
        )
        pending: deque[dict[str, Any]] = deque(
            {
                "trial_id": trial["trial_id"],
                "case_id": trial["case_id"],
                "side": trial["side"],
                "repetition": trial["repetition"],
                "attempt": 1,
            }
            for trial in checkpoint["trials"]
            if not trial["attempts"]
        )
        completed_batches = 0
        while True:
            check_directory()
            admitted = min(len(pending), options.concurrency, state.capacity())
            remaining = runner.invocation_timeout_seconds - (time.monotonic() - started)
            if not admitted or remaining <= 0:
                return stopped(
                    "complete"
                    if not pending
                    else "capacity_exhausted"
                    if not admitted
                    else "deadline"
                )
            if (
                runner.stop_after_batches is not None
                and completed_batches >= runner.stop_after_batches
            ):
                return stopped("requested")

            scheduled: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for _ in range(admitted):
                item = pending.popleft()
                check_directory()
                row = rows[item["case_id"]]
                request = _render_request(
                    plan,
                    input_text=row["input"],
                    answer=row[item["side"]],
                    reference_text=row.get("expected"),
                )
                admission = {
                    "trial_id": item["trial_id"],
                    "attempt": item["attempt"],
                    "event": _admission_event(
                        trial_id=item["trial_id"],
                        attempt=item["attempt"],
                        request=request,
                        options=options,
                    ),
                }
                if (
                    len(canonical_payload(admission["event"]))
                    > MAX_RETAINED_EVENT_BYTES
                ):
                    raise InspectJudgeError("admission event exceeds byte allowance")
                state.replace_event(item["trial_id"], admission["event"])
                write_file_no_replace(
                    _checkpoint_path(
                        runner,
                        _ADMISSION_PREFIX,
                        item["trial_id"],
                        item["attempt"],
                    ),
                    canonical_payload(admission),
                )
                check_directory()
                scheduled.append((item, request))

            async def run_item(
                item: dict[str, Any], request: dict[str, Any]
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                check_directory()
                event = await _call_one(
                    model,
                    request=request,
                    config=config,
                    options=options,
                    pacer=pacer,
                    check_directory=check_directory,
                )
                return item, event

            tasks: list[asyncio.Task[tuple[dict[str, Any], dict[str, Any]]]] = []
            for item, request in scheduled:
                tasks.append(asyncio.create_task(run_item(item, request)))

            persisted: set[tuple[str, int]] = set()

            def persist(
                item: dict[str, Any],
                event: dict[str, Any],
                persisted_set: set[tuple[str, int]] = persisted,
            ) -> None:
                identity = (item["trial_id"], item["attempt"])
                if identity in persisted_set:
                    return
                record = {
                    "trial_id": item["trial_id"],
                    "attempt": item["attempt"],
                    "event": event,
                }
                check_directory()
                state.replace_event(item["trial_id"], event)
                write_file_no_replace(
                    _checkpoint_path(
                        runner,
                        _RESULT_PREFIX,
                        item["trial_id"],
                        item["attempt"],
                    ),
                    canonical_payload(record),
                )
                check_directory()
                persisted_set.add(identity)

            async def drain(
                batch_tasks: list[
                    asyncio.Task[tuple[dict[str, Any], dict[str, Any]]]
                ] = tasks,
            ) -> None:
                for pending_task in batch_tasks:
                    if not pending_task.done():
                        pending_task.cancel()
                outcomes = await asyncio.gather(*batch_tasks, return_exceptions=True)
                for outcome in outcomes:
                    if isinstance(outcome, tuple):
                        persist(*outcome)

            try:
                async with asyncio.timeout(remaining):
                    for completed_task in asyncio.as_completed(tasks):
                        item, event = await completed_task
                        persist(item, event)
            except TimeoutError:
                await drain()
                return stopped("deadline")
            except BaseException:
                await drain()
                raise
            completed_batches += 1


async def collect(
    *,
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    runner: RunnerOptions,
    model: Any,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    on_stop: Callable[[str], None] | None = None,
) -> JudgeMeasurements:
    """Collect through a checkpoint whose directory ancestry remains pinned."""
    _check_options(plan, options)
    runner.validate()
    if options.inspect_version != INSPECT_VERSION:
        raise InspectJudgeError("live collection requires current Inspect version")
    directory = runner.checkpoint_directory.absolute()
    runner = replace(runner, checkpoint_directory=directory)
    with pinned_directory(directory, create=True) as descriptor:
        directory_stat = os.fstat(descriptor)
        if (
            directory_stat.st_uid != os.geteuid()
            or stat.S_IMODE(directory_stat.st_mode) & 0o077
        ):
            raise InspectJudgeError(
                "checkpoint directory must be caller-owned and private"
            )
        paths = (*reversed(directory.parents), directory)
        bindings = tuple(
            (path, entry_identity(path.stat(follow_symlinks=False))) for path in paths
        )
        if entry_identity(os.fstat(descriptor)) != bindings[-1][1]:
            raise PathChangedError("checkpoint directory changed while initializing")
        return await _collect_pinned(
            plan=plan,
            options=options,
            runner=runner,
            model=model,
            baseline_run=baseline_run,
            subject_run=subject_run,
            directory_fd=descriptor,
            directory_bindings=bindings,
            on_stop=on_stop,
        )
