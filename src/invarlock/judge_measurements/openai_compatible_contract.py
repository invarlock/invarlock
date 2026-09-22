"""Pure contracts shared by OpenAI-compatible collection and offline replay."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

COLLECTION_PROFILE = "openai-compatible-text-frozen-answer-v1"
SERVICES = frozenset({"vllm", "ollama", "lm_studio", "openai_compatible"})
OFFICIAL_HOSTS = frozenset(
    {
        "api.openai.com",
        "openrouter.ai",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
    }
)
COLLECTION_FIELDS = frozenset(
    {
        "profile",
        "service",
        "base_url",
        "model",
        "authentication",
        "request_timeout_seconds",
        "max_calls",
        "max_input_bytes",
        "max_output_tokens",
    }
)
BLOCKED_CREDENTIAL_FIELDS = frozenset(
    {"authorization", "api-key", "api_key", "apikey", "access_token", "secret"}
)


class OpenAICompatibleContractError(ValueError):
    """A compatible-service value violates the pure retained contract."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_base_url(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 2048
        or "\\" in value
        or any(
            ord(character) <= 32 or 127 <= ord(character) <= 159 for character in value
        )
    ):
        raise OpenAICompatibleContractError("base_url must be bounded text")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise OpenAICompatibleContractError("base_url is malformed") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise OpenAICompatibleContractError(
            "base_url must be an explicit credential-free HTTP(S) URL"
        )
    raw_host = parsed.hostname
    if "%" in raw_host:
        raise OpenAICompatibleContractError("base_url host is malformed")
    try:
        address = ipaddress.ip_address(raw_host)
    except ValueError:
        try:
            host = raw_host.rstrip(".").encode("idna").decode("ascii").casefold()
        except UnicodeError:
            raise OpenAICompatibleContractError("base_url host is malformed") from None
        if not host or any(not part for part in host.split(".")):
            raise OpenAICompatibleContractError("base_url host is malformed") from None
    else:
        host = address.compressed
    if host in OFFICIAL_HOSTS:
        raise OpenAICompatibleContractError(
            "official hosted APIs must use their qualified provider integration"
        )
    path = parsed.path.rstrip("/")
    parts = path.split("/")
    if (
        not path.endswith("/v1")
        or "%" in path
        or any(part in {".", ".."} for part in parts)
        or any(part == "" for part in parts[1:])
    ):
        raise OpenAICompatibleContractError("base_url path must end in /v1")
    if ":" in host:
        host = f"[{host}]"
    if (parsed.scheme, port) in {("http", 80), ("https", 443)}:
        port = None
    netloc = host + (f":{port}" if port is not None else "")
    return urlunsplit((parsed.scheme, netloc, path + "/", "", ""))


def normalize_configuration(
    configuration: Mapping[str, Any], *, maximum_input_bytes: int
) -> dict[str, Any]:
    fields = set(configuration) if isinstance(configuration, Mapping) else set()
    if not isinstance(configuration, Mapping) or fields not in {
        COLLECTION_FIELDS,
        COLLECTION_FIELDS | {"response_format"},
    }:
        raise OpenAICompatibleContractError(
            "OpenAI-compatible collection has unsupported fields"
        )
    result = dict(configuration)
    if result["profile"] != COLLECTION_PROFILE:
        raise OpenAICompatibleContractError(
            "unsupported OpenAI-compatible collection profile"
        )
    if not isinstance(result["service"], str) or result["service"] not in SERVICES:
        raise OpenAICompatibleContractError("unsupported compatible service family")
    if (
        not isinstance(result["model"], str)
        or not 1 <= len(result["model"]) <= 256
        or any(
            ord(character) < 33 or 127 <= ord(character) <= 159
            for character in result["model"]
        )
    ):
        raise OpenAICompatibleContractError(
            "compatible model must be explicit bounded text"
        )
    if not isinstance(result["authentication"], str) or result[
        "authentication"
    ] not in {
        "none",
        "bearer_env",
    }:
        raise OpenAICompatibleContractError(
            "unsupported compatible authentication mode"
        )
    response_format = result.get("response_format", "json_object")
    if not isinstance(response_format, str) or response_format not in {
        "json_object",
        "json_schema",
    }:
        raise OpenAICompatibleContractError("unsupported compatible response format")
    for name, maximum in (
        ("request_timeout_seconds", 3600),
        ("max_calls", 200_000),
        ("max_input_bytes", maximum_input_bytes),
        ("max_output_tokens", 10**12),
    ):
        value = result[name]
        if type(value) is not int or not 1 <= value <= maximum:
            raise OpenAICompatibleContractError(
                f"OpenAI-compatible {name} must be a bounded positive integer"
            )
    result["base_url"] = normalized_base_url(result["base_url"])
    endpoint = urlsplit(result["base_url"])
    if result["authentication"] == "bearer_env" and endpoint.scheme == "http":
        hostname = endpoint.hostname or ""
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = hostname.casefold() == "localhost"
        if not loopback:
            raise OpenAICompatibleContractError(
                "bearer authentication requires HTTPS or an explicit loopback endpoint"
            )
    return result


def service_identity(configuration: Mapping[str, Any]) -> dict[str, str]:
    identity = {
        "service": str(configuration["service"]),
        "endpoint_sha256": sha256(str(configuration["base_url"]).encode("utf-8")),
    }
    if "response_format" in configuration:
        identity["response_format"] = str(configuration["response_format"])
    return identity


def wire_request(
    *,
    response_format_profile: str,
    model: str,
    messages: Any,
    temperature: str,
    top_p: str,
    max_tokens: int,
    seed: int,
    rating_labels: list[str],
) -> dict[str, Any]:
    response_format: dict[str, Any] = {"type": "json_object"}
    if response_format_profile == "json_schema":
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "invarlock_judge_rating",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"rating": {"type": "string", "enum": rating_labels}},
                    "required": ["rating"],
                },
            },
        }
    return {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": max_tokens,
        "seed": seed,
        "n": 1,
        "stream": False,
        "response_format": response_format,
    }


def has_credential_field(value: Any) -> bool:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                if isinstance(key, str) and key.casefold() in BLOCKED_CREDENTIAL_FIELDS:
                    return True
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)
    return False


def contains_secret(value: Any, secret: str) -> bool:
    """Detect an exact secret after JSON escape decoding."""

    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                if isinstance(key, str) and secret in key:
                    return True
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, str) and secret in current:
            return True
    return False


def response_facts(
    value: Any, *, approved_models: list[str], max_output_tokens: int
) -> dict[str, Any]:
    if not isinstance(value, dict) or has_credential_field(value):
        raise OpenAICompatibleContractError(
            "compatible response is not a safe JSON object"
        )
    choices = value.get("choices")
    model = value.get("model")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or not isinstance(choices[0].get("message"), dict)
        or choices[0]["message"].get("role") != "assistant"
        or not isinstance(choices[0]["message"].get("content"), str)
        or not isinstance(model, str)
        or not 1 <= len(model) <= 256
        or any(
            ord(character) < 33 or 127 <= ord(character) <= 159 for character in model
        )
    ):
        raise OpenAICompatibleContractError(
            "compatible response lacks one bounded assistant completion and model identity"
        )
    if model not in approved_models:
        raise OpenAICompatibleContractError(
            "compatible response model is not approved by the plan"
        )
    choice = choices[0]
    if type(choice.get("index", 0)) is not int or choice.get("index", 0) != 0:
        raise OpenAICompatibleContractError(
            "compatible response choice index is invalid"
        )
    message = choice["message"]
    if (
        message.get("tool_calls") not in (None, [])
        or message.get("function_call") is not None
    ):
        raise OpenAICompatibleContractError(
            "compatible response contains an unsupported tool call"
        )
    if choice.get("finish_reason") != "stop":
        raise OpenAICompatibleContractError("compatible finish_reason is unsupported")
    request_id = value.get("id")
    fingerprint = value.get("system_fingerprint")
    for item in (request_id, fingerprint):
        if item is not None and (
            not isinstance(item, str)
            or len(item) > 256
            or any(
                ord(character) < 32 or 127 <= ord(character) <= 159
                for character in item
            )
        ):
            raise OpenAICompatibleContractError(
                "compatible response identity is invalid"
            )
    usage = value.get("usage")
    normalized_usage = None
    if usage is not None:
        if not isinstance(usage, dict):
            raise OpenAICompatibleContractError("compatible usage is invalid")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if any(
            type(item) is not int or not 0 <= item <= 10**9
            for item in (prompt_tokens, completion_tokens)
        ):
            raise OpenAICompatibleContractError("compatible usage counts are invalid")
        completion_tokens = cast(int, completion_tokens)
        if completion_tokens > max_output_tokens:
            raise OpenAICompatibleContractError(
                "compatible completion exceeds the approved output-token limit"
            )
        normalized_usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        }
    return {
        "content": message["content"],
        "model": model,
        "request_id": request_id,
        "system_fingerprint": fingerprint,
        "finish_reason": "stop",
        "usage": normalized_usage,
    }


def decode_http_blob(blob: Any) -> bytes:
    if not isinstance(blob, dict) or set(blob) != {
        "media_type",
        "encoding",
        "text",
        "sha256",
    }:
        raise OpenAICompatibleContractError(
            "retained compatible response body is invalid"
        )
    media_type = blob["media_type"]
    encoding = blob["encoding"]
    text = blob["text"]
    if (
        not isinstance(media_type, str)
        or not 1 <= len(media_type) <= 128
        or not isinstance(text, str)
        or not isinstance(encoding, str)
        or encoding not in {"utf-8", "base64"}
    ):
        raise OpenAICompatibleContractError(
            "retained compatible response body is invalid"
        )
    try:
        payload = (
            text.encode("utf-8")
            if encoding == "utf-8"
            else base64.b64decode(text, validate=True)
        )
    except (UnicodeError, ValueError):
        raise OpenAICompatibleContractError(
            "retained compatible response body is invalid"
        ) from None
    if sha256(payload) != blob["sha256"]:
        raise OpenAICompatibleContractError(
            "retained compatible response body digest is invalid"
        )
    return payload


def failure_details(outcome: str, status: int | None) -> tuple[str, str]:
    if outcome == "http_error":
        return (
            "compatible-http-error",
            f"compatible service returned HTTP {status}",
        )
    try:
        return {
            "transport_error": (
                "compatible-transport-error",
                "compatible service transport failed or exceeded its deadline",
            ),
            "malformed_response": (
                "compatible-malformed-response",
                "compatible service returned an unsupported response",
            ),
            "response_too_large": (
                "compatible-response-too-large",
                "compatible service response exceeded the retained byte limit",
            ),
            "credential_echo": (
                "compatible-credential-echo",
                "compatible service response could not be retained safely",
            ),
        }[outcome]
    except KeyError:
        raise OpenAICompatibleContractError(
            "unsupported compatible response outcome"
        ) from None


__all__ = [
    "COLLECTION_FIELDS",
    "COLLECTION_PROFILE",
    "OpenAICompatibleContractError",
    "decode_http_blob",
    "failure_details",
    "contains_secret",
    "has_credential_field",
    "normalize_configuration",
    "normalized_base_url",
    "response_facts",
    "service_identity",
    "sha256",
    "wire_request",
]
