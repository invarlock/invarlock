"""Bounded judge collection through an explicit OpenAI-compatible service."""

from __future__ import annotations

import asyncio
import base64
import importlib.metadata
import os
import stat
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.filesystem.atomic_file import write_file_no_replace
from invarlock.filesystem.paths import pinned_directory
from invarlock.judge_measurement_types import JudgeMeasurementPlan, JudgeMeasurements
from invarlock.security import network_policy_allows, temporarily_allow_network

from .contracts import (
    JUDGE_REQUEST_MAX_BYTES,
    MEASUREMENTS_FORMAT,
    MEASUREMENTS_MAX_BYTES,
    _check_attempts,
    _openai_compatible_source_errors,
    _validate_frozen_answer_bindings,
    _validate_measurement_trial_shape,
    canonical_payload,
    expected_trial_id,
    measurement_plan_digest,
    render_judge_request,
    validate_measurement_plan,
    validate_measurements,
)
from .openai_compatible_contract import (
    COLLECTION_PROFILE,
    OpenAICompatibleContractError,
    contains_secret,
    failure_details,
    has_credential_field,
    normalize_configuration,
    response_facts,
    service_identity,
    sha256,
    wire_request,
)

OPENAI_COMPATIBLE_COLLECTION_PROFILE = COLLECTION_PROFILE
OPENAI_COMPATIBLE_SOURCE_PROFILE = "retained-openai-compatible-judge-v1"
OPENAI_COMPATIBLE_SOURCE_FORMAT = "invarlock/retained-openai-compatible-judge-v1"
OPENAI_COMPATIBLE_API_KEY_ENV = "INVARLOCK_OPENAI_COMPATIBLE_API_KEY"
_MAX_HTTP_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_NEXT_SOURCE_RESERVATION_BYTES = 3 * _MAX_SOURCE_BYTES
_MEASUREMENTS_ENVELOPE_BYTES = 1024 * 1024


class OpenAICompatibleJudgeError(OpenAICompatibleContractError):
    """An explicit compatible-service request is unsupported or unsafe."""


@dataclass(frozen=True)
class OpenAICompatibleJudgeOptions:
    scorer_id: str = "judge"
    checkpoint_directory: Path | None = None

    def validate(self) -> None:
        if (
            not isinstance(self.scorer_id, str)
            or not 1 <= len(self.scorer_id) <= 128
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in self.scorer_id
            )
        ):
            raise OpenAICompatibleJudgeError(
                "OpenAI-compatible scorer_id must be a bounded printable identifier"
            )
        if self.checkpoint_directory is not None and not isinstance(
            self.checkpoint_directory, Path
        ):
            raise OpenAICompatibleJudgeError(
                "OpenAI-compatible checkpoint_directory must be a Path"
            )


def _sha256(payload: bytes) -> str:
    return sha256(payload)


def _closed_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return normalize_configuration(
            configuration, maximum_input_bytes=MEASUREMENTS_MAX_BYTES
        )
    except OpenAICompatibleContractError as exc:
        raise OpenAICompatibleJudgeError(str(exc)) from None


def openai_compatible_service_identity(
    configuration: Mapping[str, Any],
) -> dict[str, str]:
    """Return the service identity bound into an approved measurement plan."""

    checked = _closed_configuration(configuration)
    return service_identity(checked)


def _plan_constraints(
    plan: JudgeMeasurementPlan, configuration: Mapping[str, Any]
) -> None:
    judge = plan["judge"]
    config = judge["config"]
    schedule = plan["schedule"]
    if (
        judge["provider"] != "openai_compatible"
        or judge["requested_model"] != configuration["model"]
        or configuration["model"] not in judge["approved_resolved_models"]
        or judge["model_identity"] != {"kind": "hosted_api", "weights_sha256": None}
        or judge.get("service_identity") != service_identity(configuration)
    ):
        raise OpenAICompatibleJudgeError(
            "compatible service identity differs from the approved plan"
        )
    if config["reasoning_effort"] is not None or config["seed"] is None:
        raise OpenAICompatibleJudgeError(
            "compatible collection requires reasoning_effort=null and an explicit seed"
        )
    if (
        schedule["max_attempts"] != 1
        or schedule["retry_on"]
        or schedule["cache"] != "forbid"
    ):
        raise OpenAICompatibleJudgeError(
            "compatible collection requires one uncached attempt without retries"
        )


def validate_openai_compatible_collection(
    configuration: Mapping[str, Any], plan: JudgeMeasurementPlan
) -> dict[str, int]:
    """Validate the closed endpoint configuration and declared aggregate budgets."""

    validate_measurement_plan(plan)
    checked = _closed_configuration(configuration)
    _plan_constraints(plan, checked)
    expected = plan["schedule"]["expected_trials"]
    if expected > 1000:
        raise OpenAICompatibleJudgeError(
            "compatible collection exceeds the retained source-count limit"
        )
    required_output = expected * plan["judge"]["config"]["max_output_tokens"]
    if (
        checked["max_calls"] < expected
        or checked["max_output_tokens"] < required_output
    ):
        raise OpenAICompatibleJudgeError(
            "compatible collection must reserve every planned inference"
        )
    return {
        "max_calls": checked["max_calls"],
        "max_input_bytes": checked["max_input_bytes"],
        "max_output_tokens": checked["max_output_tokens"],
    }


def preflight_openai_compatible(
    configuration: Mapping[str, Any], plan: JudgeMeasurementPlan
) -> dict[str, Any]:
    """Return credential-free endpoint metadata without contacting the service."""

    budgets = validate_openai_compatible_collection(configuration, plan)
    checked = _closed_configuration(configuration)
    try:
        version = importlib.metadata.version("httpx")
    except importlib.metadata.PackageNotFoundError:
        raise OpenAICompatibleJudgeError(
            "OpenAI-compatible collection requires httpx==0.28.1"
        ) from None
    if version != "0.28.1":
        raise OpenAICompatibleJudgeError(
            "OpenAI-compatible collection requires httpx==0.28.1"
        )
    credential_available = bool(
        os.environ.get(OPENAI_COMPATIBLE_API_KEY_ENV, "").strip()
    )
    network_authorized = (
        os.environ.get("INVARLOCK_ALLOW_JUDGE_NETWORK", "").strip().lower()
        in {"1", "true", "yes", "on"}
        or network_policy_allows()
    )
    if checked["authentication"] == "bearer_env" and not credential_available:
        raise OpenAICompatibleJudgeError(
            f"{OPENAI_COMPATIBLE_API_KEY_ENV} must be set for bearer authentication"
        )
    return {
        "service": checked["service"],
        "base_url": checked["base_url"],
        "endpoint_sha256": _sha256(checked["base_url"].encode("utf-8")),
        "model": checked["model"],
        "authentication": checked["authentication"],
        "response_format": checked.get("response_format", "json_object"),
        "credential_variable": (
            OPENAI_COMPATIBLE_API_KEY_ENV
            if checked["authentication"] == "bearer_env"
            else None
        ),
        "credential_available": credential_available,
        "network_authorized": network_authorized,
        "network_authorization_variable": "INVARLOCK_ALLOW_JUDGE_NETWORK",
        "network_contacted": False,
        "budgets": budgets,
    }


def _request_body(
    plan: JudgeMeasurementPlan,
    request: bytes,
    model: str,
    response_format_profile: str,
) -> dict[str, Any]:
    normalized = parse_json_bytes(request, label="normalized compatible judge request")
    if not isinstance(
        normalized, dict
    ):  # pragma: no cover - renderer guarantees object
        raise OpenAICompatibleJudgeError(
            "normalized compatible request must be an object"
        )
    config = plan["judge"]["config"]
    seed = config["seed"]
    if seed is None:  # validated by the strict endpoint profile
        raise OpenAICompatibleJudgeError(
            "compatible collection requires an explicit seed"
        )
    return wire_request(
        response_format_profile=response_format_profile,
        model=model,
        messages=normalized["messages"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_output_tokens"],
        seed=seed,
        rating_labels=[item["label"] for item in plan["scale"]["ratings"]],
    )


def _safe_headers(response: Any, credential: str | None) -> dict[str, str]:
    retained: dict[str, str] = {}
    for name in ("content-type", "server", "x-request-id"):
        value = response.headers.get(name)
        if value is not None:
            if not isinstance(value, str):
                value = str(value)
            if credential is not None and credential in value:
                raise OpenAICompatibleJudgeError(
                    "compatible service echoed the bearer credential"
                )
            if len(value) > 512 or any(
                ord(character) < 32 or 127 <= ord(character) <= 159
                for character in value
            ):
                raise OpenAICompatibleJudgeError(
                    "compatible response contains an invalid retained header"
                )
            retained[name] = value
    return retained


def _response_facts(value: Any, *, plan: JudgeMeasurementPlan) -> dict[str, Any]:
    try:
        return response_facts(
            value,
            approved_models=plan["judge"]["approved_resolved_models"],
            max_output_tokens=plan["judge"]["config"]["max_output_tokens"],
        )
    except OpenAICompatibleContractError as exc:
        raise OpenAICompatibleJudgeError(str(exc)) from None


def _blob(payload: bytes) -> dict[str, Any]:
    return {
        "media_type": "application/json",
        "text": payload.decode("utf-8"),
        "sha256": _sha256(payload),
    }


def _http_blob(payload: bytes, media_type: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        encoding = "base64"
        text = base64.b64encode(payload).decode("ascii")
    else:
        encoding = "utf-8"
    return {
        "media_type": media_type[:128],
        "encoding": encoding,
        "text": text,
        "sha256": _sha256(payload),
    }


def _parse_rating(output: str, plan: JudgeMeasurementPlan) -> dict[str, str | None]:
    try:
        value = parse_json_bytes(
            output.encode("utf-8"), label="compatible judge completion"
        )
    except ValueError:
        value = None
    ratings = {item["label"]: item["value"] for item in plan["scale"]["ratings"]}
    if (
        isinstance(value, dict)
        and set(value) == {"rating"}
        and isinstance(value["rating"], str)
        and value["rating"] in ratings
    ):
        rating = value["rating"]
        return {"status": "ok", "rating": rating, "value": ratings[rating]}
    return {"status": "invalid", "rating": None, "value": None}


def _checkpoint_identity(
    plan_digest: str,
    configuration: Mapping[str, Any],
    options: OpenAICompatibleJudgeOptions,
) -> bytes:
    return canonical_payload(
        {
            "format": "invarlock/openai-compatible-judge-checkpoint-v1",
            "plan_sha256": plan_digest,
            "configuration": dict(configuration),
            "scorer_id": options.scorer_id,
        }
    )


def _checkpoint_sources(
    options: OpenAICompatibleJudgeOptions, identity: bytes, count: int
) -> list[dict[str, Any]]:
    directory = options.checkpoint_directory
    if directory is None:
        return []
    try:
        with pinned_directory(directory, create=True) as descriptor:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or metadata.st_uid != os.geteuid()
            ):
                raise OpenAICompatibleJudgeError(
                    "compatible checkpoint directory must be caller-owned and private (0700)"
                )
            return _checkpoint_sources_pinned(directory, identity, count)
    except OpenAICompatibleJudgeError:
        raise
    except OSError as exc:
        raise OpenAICompatibleJudgeError(
            "compatible checkpoint directory is unavailable or unsafe"
        ) from exc


def _checkpoint_sources_pinned(
    directory: Path, identity: bytes, count: int
) -> list[dict[str, Any]]:
    identity_path = directory / "identity.json"
    if identity_path.exists():
        observed = read_regular_file_bytes(
            identity_path, label="compatible checkpoint identity", max_bytes=1024 * 1024
        )
        if observed != identity:
            raise OpenAICompatibleJudgeError(
                "compatible checkpoint belongs to a different collection"
            )
    else:
        write_file_no_replace(identity_path, identity)
    allowed = {"identity.json"}
    for index in range(count):
        suffix = f"{index + 1:06d}.json"
        allowed.update({f"admission-{suffix}", f"result-{suffix}"})
    if {entry.name for entry in directory.iterdir()} - allowed:
        raise OpenAICompatibleJudgeError(
            "compatible checkpoint has an unexpected entry"
        )
    completed: list[dict[str, Any]] = []
    gap = False
    for index in range(count):
        suffix = f"{index + 1:06d}.json"
        admission_path = directory / f"admission-{suffix}"
        result_path = directory / f"result-{suffix}"
        if gap and (admission_path.exists() or result_path.exists()):
            raise OpenAICompatibleJudgeError(
                "compatible checkpoint contains a non-contiguous shard"
            )
        if result_path.exists():
            if not admission_path.exists():
                raise OpenAICompatibleJudgeError(
                    "compatible checkpoint result lacks admission"
                )
            admission = read_regular_file_bytes(
                admission_path, label="compatible checkpoint admission", max_bytes=1024
            )
            if admission != canonical_payload(
                {
                    "format": "invarlock/openai-compatible-judge-admission-v1",
                    "batch": index + 1,
                }
            ):
                raise OpenAICompatibleJudgeError(
                    "compatible checkpoint admission is invalid"
                )
            raw = read_regular_file_bytes(
                result_path,
                label="compatible checkpoint result",
                max_bytes=_MAX_SOURCE_BYTES,
            )
            value = parse_json_bytes(raw, label="compatible checkpoint result")
            if not isinstance(value, dict) or canonical_payload(value) != raw:
                raise OpenAICompatibleJudgeError(
                    "compatible checkpoint result must be a canonical object"
                )
            completed.append(cast(dict[str, Any], value))
        elif admission_path.exists():
            raise OpenAICompatibleJudgeError(
                "compatible inference was admitted without a retained result and cannot be retried"
            )
        else:
            gap = True
    return completed


def _admit(options: OpenAICompatibleJudgeOptions, index: int) -> None:
    if options.checkpoint_directory is None:
        return
    write_file_no_replace(
        options.checkpoint_directory / f"admission-{index + 1:06d}.json",
        canonical_payload(
            {
                "format": "invarlock/openai-compatible-judge-admission-v1",
                "batch": index + 1,
            }
        ),
    )


def _retain(
    options: OpenAICompatibleJudgeOptions, index: int, source: dict[str, Any]
) -> None:
    if options.checkpoint_directory is None:
        return
    write_file_no_replace(
        options.checkpoint_directory / f"result-{index + 1:06d}.json",
        canonical_payload(source),
    )


def _source_id(index: int, count: int) -> str:
    return (
        "openai-compatible-judge"
        if count == 1
        else f"openai-compatible-judge-{index + 1:06d}"
    )


def _assemble(
    plan: JudgeMeasurementPlan, plan_digest: str, sources: list[dict[str, Any]]
) -> JudgeMeasurements:
    retained = []
    trials = []
    for source in sources:
        payload = canonical_payload(source)
        trial = source["trials"][0]
        source_id = trial["attempts"][0]["source"]["source_id"]
        retained.append(
            {
                "source_id": source_id,
                "profile": OPENAI_COMPATIBLE_SOURCE_PROFILE,
                "encoding": "utf-8",
                "byte_size": len(payload),
                "media_type": "application/json",
                "content": payload.decode("utf-8"),
                "sha256": _sha256(payload),
            }
        )
        trials.append(trial)
    completed = sum(trial["status"] == "complete" for trial in trials)
    return cast(
        JudgeMeasurements,
        {
            "format": MEASUREMENTS_FORMAT,
            "profile_id": plan["profile_id"],
            "plan_sha256": plan_digest,
            "source_profile": OPENAI_COMPATIBLE_SOURCE_PROFILE,
            "sources": retained,
            "trials": trials,
            "completeness": {
                "status": "complete" if completed == len(trials) else "incomplete",
                "expected_trials": plan["schedule"]["expected_trials"],
                "recorded_trials": len(trials),
                "completed_trials": completed,
            },
        },
    )


def _validate_cached(
    sources: list[dict[str, Any]],
    *,
    pending: list[tuple[dict[str, Any], bytes, dict[str, Any]]],
    plan: JudgeMeasurementPlan,
    configuration: Mapping[str, Any],
    options: OpenAICompatibleJudgeOptions,
) -> None:
    response_ids: set[str] = set()
    for index, source in enumerate(sources):
        errors = _openai_compatible_source_errors(source, cast(dict[str, Any], plan))
        if errors:
            raise OpenAICompatibleJudgeError(errors[0])
        if source.get("collection") != dict(configuration):
            raise OpenAICompatibleJudgeError(
                "compatible checkpoint configuration differs from the current request"
            )
        trials = source.get("trials")
        if (
            not isinstance(trials, list)
            or len(trials) != 1
            or not isinstance(trials[0], dict)
        ):
            raise OpenAICompatibleJudgeError(
                "compatible checkpoint must contain one retained trial"
            )
        trial = cast(dict[str, Any], trials[0])
        _validate_measurement_trial_shape(trial)
        expected, request, _ = pending[index]
        if any(trial.get(name) != value for name, value in expected.items()):
            raise OpenAICompatibleJudgeError(
                "compatible checkpoint trial differs from its planned slot"
            )
        source_id = _source_id(index, len(pending))
        _check_attempts(
            trial,
            plan=cast(dict[str, Any], plan),
            source_ids={source_id},
            expected_request_sha256=_sha256(request),
        )
        if trial["attempts"][0]["source"] != {
            "source_id": source_id,
            "scorer_id": options.scorer_id,
            "model_event_id": trial["trial_id"],
            "record_index": 0,
            "attempt_index": 0,
        }:
            raise OpenAICompatibleJudgeError(
                "compatible checkpoint source mapping is invalid"
            )
        response_id = trial["attempts"][0]["request_id"]
        if response_id is not None:
            if response_id in response_ids:
                raise OpenAICompatibleJudgeError(
                    "compatible checkpoint reused a non-null response ID"
                )
            response_ids.add(response_id)


def _client(configuration: Mapping[str, Any], headers: Mapping[str, str]) -> Any:
    import httpx

    return httpx.AsyncClient(
        base_url=configuration["base_url"],
        headers=dict(headers),
        timeout=None,
        follow_redirects=False,
        trust_env=False,
    )


async def _bounded_exchange(
    client: Any,
    *,
    body: bytes,
    timeout_seconds: int,
    credential: str | None,
) -> dict[str, Any]:
    """Stream one bounded response under a monotonic whole-call deadline."""

    async with asyncio.timeout(timeout_seconds):
        async with client.stream("POST", "chat/completions", content=body) as response:
            status = response.status_code
            content_encoding = response.headers.get("content-encoding")
            if (
                content_encoding is not None
                and content_encoding.casefold() != "identity"
            ):
                return {
                    "outcome": "malformed_response",
                    "response_status": status,
                    "response_headers": {},
                    "response_body": None,
                    "raw": None,
                }
            try:
                headers = _safe_headers(response, credential)
            except OpenAICompatibleJudgeError as exc:
                if "bearer credential" in str(exc):
                    return {
                        "outcome": "credential_echo",
                        "response_status": status,
                        "response_headers": {},
                        "response_body": None,
                        "raw": None,
                    }
                return {
                    "outcome": "malformed_response",
                    "response_status": status,
                    "response_headers": {},
                    "response_body": None,
                    "raw": None,
                }
            if 300 <= status <= 399:
                return {
                    "outcome": "http_error",
                    "response_status": status,
                    "response_headers": headers,
                    "response_body": None,
                    "raw": None,
                }
            chunks: list[bytes] = []
            size = 0
            secret = credential.encode("utf-8") if credential is not None else None
            async for chunk in response.aiter_raw(chunk_size=64 * 1024):
                size += len(chunk)
                if size > _MAX_HTTP_RESPONSE_BYTES:
                    return {
                        "outcome": "response_too_large",
                        "response_status": status,
                        "response_headers": headers,
                        "response_body": None,
                        "raw": None,
                    }
                chunks.append(chunk)
            raw = b"".join(chunks)
            if secret is not None and secret in raw:
                return {
                    "outcome": "credential_echo",
                    "response_status": status,
                    "response_headers": {},
                    "response_body": None,
                    "raw": None,
                }
            try:
                decoded_for_safety = parse_json_bytes(
                    raw, label="compatible HTTP response"
                )
            except ValueError:
                if credential is not None:
                    return {
                        "outcome": "malformed_response",
                        "response_status": status,
                        "response_headers": headers,
                        "response_body": None,
                        "raw": None,
                    }
                decoded_for_safety = None
            if has_credential_field(decoded_for_safety) or (
                credential is not None
                and contains_secret(decoded_for_safety, credential)
            ):
                return {
                    "outcome": "credential_echo",
                    "response_status": status,
                    "response_headers": {},
                    "response_body": None,
                    "raw": None,
                }
            media_type = headers.get("content-type", "application/octet-stream")
            return {
                "outcome": "success" if status == 200 else "http_error",
                "response_status": status,
                "response_headers": headers,
                "response_body": _http_blob(raw, media_type),
                "raw": raw,
            }


async def _exchange_once(
    client: Any,
    *,
    body: bytes,
    credential: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    async with client:
        return await _bounded_exchange(
            client,
            body=body,
            timeout_seconds=timeout_seconds,
            credential=credential,
        )


def _failure_details(outcome: str, status: int | None) -> tuple[str, str]:
    return failure_details(outcome, status)


def collect_openai_compatible(
    plan: JudgeMeasurementPlan,
    *,
    configuration: Mapping[str, Any],
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    options: OpenAICompatibleJudgeOptions = OpenAICompatibleJudgeOptions(),
) -> JudgeMeasurements:
    """Collect exact chat-completion outcomes from one explicit compatible endpoint."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise OpenAICompatibleJudgeError(
            "synchronous OpenAI-compatible collection cannot run inside an active "
            "event loop; use a worker thread or the CLI"
        )
    options.validate()
    checked = _closed_configuration(configuration)
    preflight_openai_compatible(checked, plan)
    _validate_frozen_answer_bindings(plan, baseline_run, subject_run)
    allow_network = os.environ.get(
        "INVARLOCK_ALLOW_JUDGE_NETWORK", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not allow_network and not network_policy_allows():
        raise OpenAICompatibleJudgeError(
            "compatible judge collection requires an allowed network policy; set "
            "INVARLOCK_ALLOW_JUDGE_NETWORK=1 only for collection"
        )
    plan_digest = measurement_plan_digest(plan)
    rows = {
        "baseline": {row["id"]: row for row in baseline_run["records"]},
        "subject": {row["id"]: row for row in subject_run["records"]},
    }
    pending: list[tuple[dict[str, Any], bytes, dict[str, Any]]] = []
    input_bytes = 0
    for binding in sorted(plan["answer_bindings"], key=lambda item: item["case_id"]):
        case_id = binding["case_id"]
        for side in ("baseline", "subject"):
            row = rows[side][case_id]
            for repetition in range(1, plan["schedule"]["repetitions"] + 1):
                request = render_judge_request(
                    plan,
                    input_text=row["input"],
                    answer_text=row["output"],
                    reference_text=row["expected"],
                )
                body = _request_body(
                    plan,
                    request,
                    checked["model"],
                    checked.get("response_format", "json_object"),
                )
                body_bytes = canonical_payload(body)
                if len(body_bytes) > JUDGE_REQUEST_MAX_BYTES:
                    raise OpenAICompatibleJudgeError(
                        "compatible wire request exceeds the per-call byte limit"
                    )
                input_bytes += len(body_bytes)
                if input_bytes > checked["max_input_bytes"]:
                    raise OpenAICompatibleJudgeError(
                        "compatible rendered requests exceed the aggregate "
                        "input-byte reservation"
                    )
                trial_id = expected_trial_id(plan_digest, case_id, side, repetition)
                pending.append(
                    (
                        {
                            "trial_id": trial_id,
                            "case_id": case_id,
                            "side": side,
                            "repetition": repetition,
                            "answer_sha256": cast(dict[str, Any], binding)[
                                f"{side}_answer_sha256"
                            ],
                            "plan_sha256": plan_digest,
                        },
                        request,
                        body,
                    )
                )
    if (
        _MEASUREMENTS_ENVELOPE_BYTES + len(pending) * _NEXT_SOURCE_RESERVATION_BYTES
        > MEASUREMENTS_MAX_BYTES
        and options.checkpoint_directory is None
    ):
        raise OpenAICompatibleJudgeError(
            "large compatible schedules require a durable checkpoint directory"
        )
    identity = _checkpoint_identity(plan_digest, checked, options)
    sources = _checkpoint_sources(options, identity, len(pending))
    _validate_cached(
        sources,
        pending=pending,
        plan=plan,
        configuration=checked,
        options=options,
    )
    headers = {
        "accept-encoding": "identity",
        "content-type": "application/json",
    }
    credential = None
    if checked["authentication"] == "bearer_env":
        credential = os.environ.get(OPENAI_COMPATIBLE_API_KEY_ENV, "").strip()
        if not credential:
            raise OpenAICompatibleJudgeError(
                f"{OPENAI_COMPATIBLE_API_KEY_ENV} must be set for bearer authentication"
            )
        headers["authorization"] = f"Bearer {credential}"
    scope = temporarily_allow_network() if allow_network else nullcontext()
    with scope:
        for index in range(len(sources), len(pending)):
            current = _assemble(plan, plan_digest, sources) if sources else None
            retained_size = (
                len(canonical_payload(current)) if current is not None else 0
            )
            if (
                retained_size
                + _NEXT_SOURCE_RESERVATION_BYTES
                + _MEASUREMENTS_ENVELOPE_BYTES
                > MEASUREMENTS_MAX_BYTES
            ):
                raise OpenAICompatibleJudgeError(
                    "compatible retained capacity was exhausted before the next admission"
                )
            trial, normalized_request, body = pending[index]
            source_id = _source_id(index, len(pending))
            try:
                client = _client(checked, headers)
            except Exception:
                raise OpenAICompatibleJudgeError(
                    "OpenAI-compatible HTTP client could not be initialized"
                ) from None
            _admit(options, index)
            try:
                exchange = asyncio.run(
                    _exchange_once(
                        client,
                        body=canonical_payload(body),
                        credential=credential,
                        timeout_seconds=checked["request_timeout_seconds"],
                    )
                )
            except Exception:
                exchange = {
                    "outcome": "transport_error",
                    "response_status": None,
                    "response_headers": {},
                    "response_body": None,
                    "raw": None,
                }
            service_identity: dict[str, Any] = {
                "service": checked["service"],
                "endpoint_sha256": _sha256(checked["base_url"].encode("utf-8")),
                "requested_model": checked["model"],
                "response_model": None,
                "request_id": None,
                "system_fingerprint": None,
            }
            outcome = exchange["outcome"]
            facts = None
            if outcome == "success":
                raw = cast(bytes, exchange["raw"])
                try:
                    decoded_text = raw.decode("utf-8", errors="strict")
                    decoded = parse_json_bytes(
                        decoded_text.encode("utf-8"),
                        label="compatible chat completion",
                    )
                    facts = _response_facts(decoded, plan=plan)
                    if len(facts["content"].encode("utf-8")) > JUDGE_REQUEST_MAX_BYTES:
                        raise OpenAICompatibleJudgeError(
                            "compatible completion exceeds the retained byte limit"
                        )
                except (UnicodeError, ValueError):
                    outcome = "malformed_response"
            if outcome == "success" and facts is not None:
                output = facts["content"]
                parsed = _parse_rating(output, plan)
                attempt_response = _blob(output.encode("utf-8"))
                attempt_status = "completed"
                attempt_error = None
                selected = 1
                service_identity.update(
                    response_model=facts["model"],
                    request_id=facts["request_id"],
                    system_fingerprint=facts["system_fingerprint"],
                )
                usage = facts["usage"]
                finish_reason = facts["finish_reason"]
                resolved_model = facts["model"]
            else:
                parsed = {"status": "unavailable", "rating": None, "value": None}
                attempt_response = None
                attempt_status = "cancelled"
                error_code, error_message = _failure_details(
                    outcome, exchange["response_status"]
                )
                attempt_error = {
                    "code": error_code,
                    "message": error_message,
                }
                selected = None
                usage = None
                finish_reason = None
                resolved_model = None
            attempt = {
                "attempt": 1,
                "role": "judge",
                "resolved_model": resolved_model,
                "status": attempt_status,
                "request": _blob(normalized_request),
                "response": attempt_response,
                "request_id": service_identity["request_id"],
                "finish_reason": finish_reason,
                "error": attempt_error,
                "usage": usage,
                "cache": "none",
                "source": {
                    "source_id": source_id,
                    "scorer_id": options.scorer_id,
                    "model_event_id": trial["trial_id"],
                    "record_index": 0,
                    "attempt_index": 0,
                },
            }
            trial.update(
                status="complete" if parsed["status"] == "ok" else "incomplete",
                attempts=[attempt],
                selected_attempt=selected,
                parse=parsed,
            )
            source = {
                "format": OPENAI_COMPATIBLE_SOURCE_FORMAT,
                "collection": checked,
                "http": {
                    "outcome": outcome,
                    "request": body,
                    "response_status": exchange["response_status"],
                    "response_headers": exchange["response_headers"],
                    "response_body": exchange["response_body"],
                },
                "service_identity": service_identity,
                "trials": [trial],
            }
            encoded = canonical_payload(source)
            if len(encoded) > _MAX_SOURCE_BYTES:
                raise OpenAICompatibleJudgeError(
                    "compatible retained source exceeds the per-source byte limit"
                )
            _retain(options, index, source)
            sources.append(source)
    result = _assemble(plan, plan_digest, sources)
    if len(canonical_payload(result)) > MEASUREMENTS_MAX_BYTES:
        raise OpenAICompatibleJudgeError(
            "compatible retained measurements exceed the aggregate byte limit"
        )
    validate_measurements(
        result,
        plan,
        baseline_run=baseline_run,
        subject_run=subject_run,
    )
    return result


__all__ = [
    "OPENAI_COMPATIBLE_API_KEY_ENV",
    "OPENAI_COMPATIBLE_COLLECTION_PROFILE",
    "OPENAI_COMPATIBLE_SOURCE_FORMAT",
    "OPENAI_COMPATIBLE_SOURCE_PROFILE",
    "OpenAICompatibleJudgeError",
    "OpenAICompatibleJudgeOptions",
    "collect_openai_compatible",
    "openai_compatible_service_identity",
    "preflight_openai_compatible",
    "validate_openai_compatible_collection",
]
