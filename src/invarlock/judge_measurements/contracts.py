"""Semantic validation for retained bounded judge measurements.

JSON Schema closes each wire shape.  This module validates relationships that
schemas cannot express: content digests, fixed schedules, frozen-answer
bindings, retry selection, source replay, and completeness accounting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, NoReturn, cast

from jsonschema import Draft202012Validator

from invarlock.core.runtime_provider import (
    EvaluationInputPart,
    ModelRuntimeSpec,
    artifact_identity_sha256,
    evaluation_input_parts_sha256,
)
from invarlock.core.runtime_provider.behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.evaluation_comparison.comparison import _check_run
from invarlock.evaluation_records.cases import validate_run_case_set
from invarlock.evaluation_records.io import run_digest
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.judge_measurement_types import (
    JudgeMeasurementPlan,
    JudgeMeasurements,
)
from invarlock.judge_measurements.openai_compatible_contract import (
    OpenAICompatibleContractError,
    decode_http_blob,
    failure_details,
    has_credential_field,
    normalize_configuration,
    response_facts,
    service_identity,
    wire_request,
)
from invarlock.public_contracts import (
    load_judge_measurement_plan_schema,
    load_judge_measurements_schema,
)
from invarlock.runtime_provider_evidence import (
    RuntimeProviderEvidenceError,
    decode_artifact_identity,
    decode_runtime_provider_receipt,
    decode_scoring_observation,
    runtime_provider_evidence_errors,
    runtime_request_binding_errors,
)

PLAN_MAX_BYTES = 64 * 1024 * 1024
MEASUREMENTS_MAX_BYTES = 384 * 1024 * 1024
PLAN_FORMAT = "invarlock/judge-measurement-plan-v1"
MEASUREMENTS_FORMAT = "invarlock/judge-measurements-v1"
SOURCE_FORMAT = "invarlock/retained-judge-json-v1"
INSPECT_SOURCE_FORMAT = "invarlock/retained-inspect-model-events-v1"
RUNTIME_PROVIDER_SOURCE_FORMAT = "retained-runtime-provider-judge-v1"
OPENAI_COMPATIBLE_SOURCE_PROFILE = "retained-openai-compatible-judge-v1"
OPENAI_COMPATIBLE_SOURCE_FORMAT = "invarlock/retained-openai-compatible-judge-v1"
TRIAL_ID_SCHEME = "plan-case-side-repetition-sha256-v1"
JUDGE_REQUEST_MAX_BYTES = 1024 * 1024
_INSPECT_0_3_263_REASONING_EFFORTS = {
    "openai/gpt-5.6-sol": {"none", "low", "medium", "high", "xhigh", "max"},
    "openai/gpt-5.6-luna": {"none", "low", "medium", "high", "xhigh", "max"},
}
_INSPECT_BRIDGED_REASONING_TOKENS = {
    "minimal": 2048,
    "low": 4096,
    "medium": 10000,
    "high": 16000,
    "xhigh": 32000,
    "max": 32000,
}


class JudgeMeasurementContractError(ValueError):
    """A judge plan or retained-measurement relationship is invalid."""


def _fail(message: str) -> NoReturn:
    raise JudgeMeasurementContractError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(text: str) -> str:
    return _sha256(text.encode("utf-8"))


def canonical_payload(value: object) -> bytes:
    """Return the compact canonical bytes used by judge contract digests."""

    return canonical_json_bytes(value, newline=False)


def _bounded_canonical_payload(value: object, maximum: int, label: str) -> bytes:
    """Serialize canonically while refusing growth beyond ``maximum`` bytes."""

    encoder = json.JSONEncoder(
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    result = bytearray()
    try:
        for chunk in encoder.iterencode(value):
            encoded = chunk.encode("utf-8")
            if len(result) + len(encoded) > maximum:
                _fail(f"{label} exceeds the {maximum}-byte limit")
            result.extend(encoded)
    except JudgeMeasurementContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise JudgeMeasurementContractError(f"{label} is not canonical JSON") from exc
    return bytes(result)


def measurement_plan_digest(plan: JudgeMeasurementPlan) -> str:
    """Return the canonical SHA-256 digest of one validated plan value."""

    validate_measurement_plan(plan)
    return _sha256(canonical_payload(plan))


def expected_trial_id(
    plan_sha256: str, case_id: str, side: str, repetition: int
) -> str:
    """Derive the stable bounded identifier for one planned trial slot."""

    material = canonical_payload([plan_sha256, case_id, side, repetition])
    return "trial-" + _sha256(material)


def render_judge_request(
    plan: JudgeMeasurementPlan,
    *,
    input_text: object,
    answer_text: object,
    reference_text: object = None,
) -> bytes:
    """Render the closed normalized request used by the first judge profile."""

    if not isinstance(input_text, str) or not isinstance(answer_text, str):
        _fail("the text-frozen-answer profile requires string inputs and answers")
    checked_input = input_text
    checked_answer = answer_text
    prompt = plan["prompt"]
    reference_mode = prompt.get("reference_mode", "none")
    if reference_mode not in ("none", "per_case"):
        _fail("unsupported judge reference mode")
    if reference_mode == "per_case":
        if not isinstance(reference_text, str):
            _fail("per-case judging requires a string reference")
        _bounded_canonical_payload(
            reference_text, JUDGE_REQUEST_MAX_BYTES, "per-case judge reference"
        )
    references = [
        {"id": item["id"], "text": item["text"]} for item in prompt["references"]
    ]

    content_bytes = len(prompt["system"].encode("utf-8"))

    def user_content(
        input_value: str, answer_value: str, *, include_reference: bool = False
    ) -> str:
        nonlocal content_bytes
        encoded = _bounded_canonical_payload(
            {
                "answer": answer_value,
                "input": input_value,
                "instruction": prompt["template"],
                "references": references,
                "rubric": plan["rubric"]["text"],
                **({"reference": reference_text} if include_reference else {}),
            },
            JUDGE_REQUEST_MAX_BYTES,
            "normalized judge request",
        )
        content_bytes += len(encoded)
        if content_bytes > JUDGE_REQUEST_MAX_BYTES:
            _fail(
                f"normalized judge request exceeds the {JUDGE_REQUEST_MAX_BYTES}-byte limit"
            )
        return encoded.decode("utf-8")

    messages: list[dict[str, str]] = []
    if prompt["system"]:
        messages.append({"role": "system", "content": prompt["system"]})
    for demonstration in prompt["demonstrations"]:
        assistant = _bounded_canonical_payload(
            {"rating": demonstration["rating"]},
            JUDGE_REQUEST_MAX_BYTES,
            "normalized judge request",
        ).decode("utf-8")
        content_bytes += len(assistant.encode("utf-8"))
        if content_bytes > JUDGE_REQUEST_MAX_BYTES:
            _fail(
                f"normalized judge request exceeds the {JUDGE_REQUEST_MAX_BYTES}-byte limit"
            )
        messages.extend(
            (
                {
                    "role": "user",
                    "content": user_content(
                        demonstration["input"], demonstration["answer"]
                    ),
                },
                {
                    "role": "assistant",
                    "content": assistant,
                },
            )
        )
    messages.append(
        {
            "role": "user",
            "content": user_content(
                checked_input,
                checked_answer,
                include_reference=reference_mode == "per_case",
            ),
        }
    )
    return _bounded_canonical_payload(
        {
            "config": plan["judge"]["config"],
            "format": "invarlock/judge-request-v1",
            "messages": messages,
            "model": plan["judge"]["requested_model"],
            "model_role": "judge",
            "response_format": {
                "additional_properties": False,
                "rating_labels": [
                    rating["label"] for rating in plan["scale"]["ratings"]
                ],
                "required": ["rating"],
                "type": "json_object",
            },
            "tools": [],
        },
        JUDGE_REQUEST_MAX_BYTES,
        "normalized judge request",
    )


def render_runtime_prompt(
    plan: JudgeMeasurementPlan, normalized_request: bytes
) -> bytes:
    """Render one plan-bound direct-runtime prompt without tokenizer execution."""

    runtime_format = plan["prompt"].get("runtime_format", "canonical-json-v1")
    if runtime_format == "canonical-json-v1":
        return normalized_request
    if runtime_format != "chatml-v1":
        _fail("unsupported direct judge runtime prompt format")
    try:
        request = parse_json_bytes(
            normalized_request, label="normalized direct judge request"
        )
    except StrictJsonError as exc:
        raise JudgeMeasurementContractError(str(exc)) from exc
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        _fail("normalized direct judge request must contain messages")
    messages = request["messages"]
    if not messages or not isinstance(messages[-1], dict):
        _fail("normalized direct judge request must end with a user message")
    reserved = ("<|im_start|>", "<|im_end|>", "<|im_sep|>")
    rendered: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            _fail("normalized direct judge messages have an unsupported shape")
        role = message["role"]
        content = message["content"]
        if (
            not isinstance(role, str)
            or role not in {"system", "user", "assistant"}
            or not isinstance(content, str)
        ):
            _fail("normalized direct judge messages require closed text roles")
        if any(marker in content for marker in reserved):
            _fail("direct judge prompt content contains a reserved ChatML delimiter")
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if messages[-1]["role"] != "user":
        _fail("normalized direct judge request must end with a user message")
    rendered.append("<|im_start|>assistant\n")
    payload = "".join(rendered).encode("utf-8")
    if len(payload) > JUDGE_REQUEST_MAX_BYTES:
        _fail(f"direct judge prompt exceeds the {JUDGE_REQUEST_MAX_BYTES}-byte limit")
    return payload


@lru_cache(maxsize=2)
def _validator(kind: str) -> Draft202012Validator:
    if kind == "plan":
        return Draft202012Validator(load_judge_measurement_plan_schema())
    if kind == "measurements":
        return Draft202012Validator(load_judge_measurements_schema())
    raise AssertionError(f"unknown validator kind: {kind}")


def _validate_schema(value: dict[str, Any], kind: str) -> None:
    error = next(_validator(kind).iter_errors(value), None)
    if error is not None:
        path = "/".join(str(part) for part in error.absolute_path)
        location = f" at {path}" if path else ""
        _fail(f"judge {kind} contract is invalid{location}: {error.message[:240]}")


def _bounded_canonical(value: object, maximum: int, label: str) -> None:
    _bounded_canonical_payload(value, maximum, label)


def _precheck_plan_counts(raw: dict[str, Any]) -> None:
    sampling = raw.get("sampling")
    bindings = raw.get("answer_bindings")
    if isinstance(sampling, dict):
        cases = sampling.get("case_units")
        if isinstance(cases, list) and len(cases) > 10_000:
            _fail("judge measurement plan exceeds the case limit")
    if isinstance(bindings, list) and len(bindings) > 10_000:
        _fail("judge measurement plan exceeds the answer-binding limit")


def _precheck_measurement_counts(raw: dict[str, Any]) -> None:
    sources = raw.get("sources")
    trials = raw.get("trials")
    if isinstance(sources, list) and len(sources) > 1_000:
        _fail("judge measurements exceed the source-count limit")
    if isinstance(trials, list):
        if len(trials) > 200_000:
            _fail("judge measurements exceed the trial-count limit")
        for trial in trials:
            if not isinstance(trial, dict):
                continue
            attempts = trial.get("attempts")
            if isinstance(attempts, list) and len(attempts) > 3:
                _fail("judge measurements exceed the per-trial attempt limit")


def _require_integer(value: Any, label: str) -> None:
    if type(value) is not int:
        _fail(f"{label} must be an integer")


def _validate_inspect_grader(value: object) -> None:
    if (
        not isinstance(value, str)
        or not 0 < len(value) <= 256
        or any(part in value for part in (":", "@", "?", "#", "\\"))
        or any(ord(char) < 33 for char in value)
    ):
        _fail("grader must be an explicit model identity without URL credentials")


def _validate_inspect_plan_collection_identity(
    plan: JudgeMeasurementPlan, *, grader: object, inspect_version: object
) -> None:
    _validate_inspect_grader(grader)
    if grader != plan["judge"]["requested_model"]:
        _fail("grader differs from approved plan")
    if plan["judge"]["model_identity"]["kind"] != "hosted_api":
        _fail("local weight execution is not qualified by this adapter")
    provider = str(grader).partition("/")[0]
    config = plan["judge"]["config"]
    if provider in {"anthropic", "google"} and config["seed"] is not None:
        _fail(f"the pinned Inspect {provider} adapter does not support seed")
    if provider == "anthropic" and Decimal(config["top_p"]) != Decimal(1):
        _fail("the configured Anthropic profile requires top_p=1")
    effort = config["reasoning_effort"]
    if provider in {"anthropic", "google"} and effort not in (None, "none"):
        model_name = str(grader).split("/", 1)[1]
        supported = (
            model_name.startswith(
                ("claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5")
            )
            if provider == "anthropic"
            else model_name.startswith("gemini-2.5-")
        )
        budget = _INSPECT_BRIDGED_REASONING_TOKENS.get(effort)
        if not supported or budget is None:
            _fail(
                f"the pinned Inspect {provider} reasoning effort is not qualified for this model"
            )
        if config["max_output_tokens"] <= budget:
            _fail(
                f"the pinned Inspect {provider} output limit must exceed its reasoning budget"
            )
    supported_efforts = _INSPECT_0_3_263_REASONING_EFFORTS.get(grader)
    if supported_efforts is not None:
        if (
            inspect_version != "0.3.263"
            or Decimal(config["temperature"]) != Decimal(1)
            or config["reasoning_effort"] not in supported_efforts
        ):
            _fail(
                f"{grader} requires Inspect 0.3.263, approved temperature 1, "
                "and a supported explicit reasoning_effort"
            )


def _check_trial_integer_types(trial: dict[str, Any]) -> None:
    _require_integer(trial.get("repetition"), "trial repetition")
    selected = trial.get("selected_attempt")
    if selected is not None:
        _require_integer(selected, "selected attempt")
    attempts = trial.get("attempts")
    if not isinstance(attempts, list):
        _fail("trial attempts must be an array")
    for attempt in attempts:
        if not isinstance(attempt, dict):
            _fail("trial attempts must be objects")
        _require_integer(attempt.get("attempt"), "attempt number")
        source = attempt.get("source")
        if not isinstance(source, dict):
            _fail("attempt source mapping must be an object")
        _require_integer(source.get("record_index"), "source record index")
        _require_integer(source.get("attempt_index"), "source attempt index")
        usage = attempt.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                _fail("attempt usage must be an object or null")
            _require_integer(usage.get("input_tokens"), "usage input_tokens")
            _require_integer(usage.get("output_tokens"), "usage output_tokens")


def _validate_measurement_trial_shape(trial: dict[str, Any]) -> None:
    schema = load_judge_measurements_schema()["properties"]["trials"]["items"]
    error = next(
        _validator("measurements").evolve(schema=schema).iter_errors(trial), None
    )
    if error is not None:
        path = "/".join(str(part) for part in error.absolute_path)
        location = f" at {path}" if path else ""
        _fail(f"retained judge trial is invalid{location}: {error.message[:240]}")
    _check_trial_integer_types(trial)


def validate_measurement_plan(value: JudgeMeasurementPlan) -> None:
    """Validate the closed plan and all cross-field scheduling invariants."""

    raw = cast(dict[str, Any], value)
    _precheck_plan_counts(raw)
    _bounded_canonical(raw, PLAN_MAX_BYTES, "judge measurement plan")
    _validate_schema(raw, "plan")

    rubric = raw["rubric"]
    if _text_sha256(rubric["text"]) != rubric["sha256"]:
        _fail("rubric digest does not match its UTF-8 text")

    reference_ids: set[str] = set()
    for reference in raw["prompt"]["references"]:
        if reference["id"] in reference_ids:
            _fail("prompt reference IDs must be unique")
        reference_ids.add(reference["id"])
        if _text_sha256(reference["text"]) != reference["sha256"]:
            _fail(f"prompt reference {reference['id']!r} digest does not match")

    ratings: dict[str, Any] = {}
    rating_values: set[Decimal] = set()
    for rating in raw["scale"]["ratings"]:
        label = rating["label"]
        numeric_value = Decimal(str(rating["value"]))
        if label in ratings:
            _fail("rating labels must be unique")
        if numeric_value in rating_values:
            _fail("rating numeric values must be unique")
        ratings[label] = rating["value"]
        rating_values.add(numeric_value)
    for demonstration in raw["prompt"]["demonstrations"]:
        if demonstration["rating"] not in ratings:
            _fail("every demonstration rating must exist in the declared scale")

    judge = raw["judge"]
    identity = judge["model_identity"]
    if identity["kind"] == "hosted_api" and identity["weights_sha256"] is not None:
        _fail("hosted API judge identity must not claim a weights digest")
    if identity["kind"] == "local_weights" and identity["weights_sha256"] is None:
        _fail("local judge identity requires a weights digest")
    service_identity = judge.get("service_identity")
    if judge["provider"] == "openai_compatible":
        if service_identity is None:
            _fail("OpenAI-compatible judge plans require a service identity")
    elif service_identity is not None:
        _fail("service identity is only valid for OpenAI-compatible judge plans")
    runtime_format = raw["prompt"].get("runtime_format")
    if runtime_format is not None and judge["provider"] not in {
        "hf_transformers",
        "llama_cpp",
    }:
        _fail("runtime prompt format is only valid for direct runtime judge plans")

    case_units: dict[str, str] = {}
    for item in raw["sampling"]["case_units"]:
        if item["case_id"] in case_units:
            _fail("sampling case IDs must be unique")
        case_units[item["case_id"]] = item["unit_id"]

    bindings: dict[str, dict[str, Any]] = {}
    for binding in raw["answer_bindings"]:
        if binding["case_id"] in bindings:
            _fail("answer-binding case IDs must be unique")
        bindings[binding["case_id"]] = binding
    if set(case_units) != set(bindings):
        _fail("sampling and answer bindings must contain the same case IDs")

    schedule = raw["schedule"]
    for field in ("repetitions", "max_attempts", "expected_trials"):
        if type(schedule[field]) is not int:
            _fail(f"schedule {field} must be an integer")
    config = raw["judge"]["config"]
    _require_integer(config["max_output_tokens"], "judge max_output_tokens")
    if config["seed"] is not None:
        _require_integer(config["seed"], "judge seed")
    if schedule.get("trial_id_scheme") != TRIAL_ID_SCHEME:
        _fail("unsupported judge trial ID scheme")
    expected = len(case_units) * 2 * schedule["repetitions"]
    if schedule["expected_trials"] != expected:
        _fail("expected_trials must equal cases × two sides × repetitions")
    if schedule["max_attempts"] > 1 and schedule["retry_on"] != ["transport_error"]:
        _fail("multiple attempts require transport_error as the sole retry condition")
    # Reject plans whose fixed prompt material already exceeds the request
    # envelope. Frozen case inputs and answers are checked when rendered.
    request = render_judge_request(
        value, input_text="", answer_text="", reference_text=""
    )
    render_runtime_prompt(value, request)


def _load_object(path: Path, *, maximum: int, label: str) -> dict[str, Any]:
    try:
        payload = read_regular_file_bytes(path, label=label, max_bytes=maximum)
        decoded = parse_json_bytes(payload, label=label)
    except StrictJsonError as exc:
        raise JudgeMeasurementContractError(str(exc)) from exc
    if not isinstance(decoded, dict):
        _fail(f"{label} must decode to a JSON object")
    return cast(dict[str, Any], decoded)


def load_measurement_plan(path: Path) -> JudgeMeasurementPlan:
    """Read and validate a bounded plan from one immutable file snapshot."""

    decoded = _load_object(
        Path(path), maximum=PLAN_MAX_BYTES, label="judge measurement plan"
    )
    plan = cast(JudgeMeasurementPlan, decoded)
    validate_measurement_plan(plan)
    return plan


def _check_blob(blob: dict[str, Any], label: str) -> None:
    content = blob["text"].encode("utf-8")
    if len(content) > JUDGE_REQUEST_MAX_BYTES:
        _fail(f"{label} exceeds the {JUDGE_REQUEST_MAX_BYTES}-byte limit")
    if hashlib.sha256(content).hexdigest() != blob["sha256"]:
        _fail(f"{label} digest does not match its UTF-8 text")


def _check_attempts(
    trial: dict[str, Any],
    *,
    plan: dict[str, Any],
    source_ids: set[str],
    expected_request_sha256: str,
) -> None:
    attempts = trial["attempts"]
    expected_numbers = list(range(1, len(attempts) + 1))
    if [attempt["attempt"] for attempt in attempts] != expected_numbers:
        _fail(f"trial {trial['trial_id']!r} attempts must be contiguous from one")
    if len(attempts) > plan["schedule"]["max_attempts"]:
        _fail(f"trial {trial['trial_id']!r} exceeds its approved attempt limit")

    completed: list[int] = []
    terminal = False
    for index, attempt in enumerate(attempts):
        if terminal:
            _fail(
                f"trial {trial['trial_id']!r} retained an attempt after a terminal result"
            )
        _check_blob(attempt["request"], "judge request")
        if attempt["request"]["sha256"] != expected_request_sha256:
            _fail(f"trial {trial['trial_id']!r} request was not approved by the plan")
        if attempt["response"] is not None:
            _check_blob(attempt["response"], "judge response")
        if attempt["source"]["source_id"] not in source_ids:
            _fail(f"trial {trial['trial_id']!r} references an unknown source")
        if attempt["source"]["attempt_index"] != index:
            _fail(f"trial {trial['trial_id']!r} source attempt index is inconsistent")
        if attempt["cache"] != "none":
            _fail("the fixed-answer judge profile forbids reused cached responses")

        status = attempt["status"]
        if status == "completed":
            completed.append(attempt["attempt"])
            if attempt["response"] is None or attempt["error"] is not None:
                _fail("completed judge attempts require a response and no error")
            if (
                attempt["resolved_model"]
                not in plan["judge"]["approved_resolved_models"]
            ):
                _fail("completed judge attempt used an unapproved resolved model")
            terminal = True
        elif status == "transport_error":
            if attempt["error"] is None:
                _fail("transport errors require retained error information")
            if attempt["response"] is not None:
                _fail("transport errors must not retain a completed response")
            if (
                index + 1 < len(attempts)
                and "transport_error" not in plan["schedule"]["retry_on"]
            ):
                _fail("transport retry was not approved by the plan")
        else:
            if attempt["error"] is None and status in {
                "timeout_ambiguous",
                "cancelled",
            }:
                _fail(f"{status} attempts require retained error information")
            terminal = True

    selected = trial["selected_attempt"]
    expected_selected = completed[0] if completed else None
    if selected != expected_selected:
        _fail(
            f"trial {trial['trial_id']!r} did not select the first completed response"
        )

    parsed = trial["parse"]
    scale = {item["label"]: item["value"] for item in plan["scale"]["ratings"]}
    if selected is not None:
        attempt = attempts[selected - 1]
        response = attempt["response"]
        assert response is not None  # established by completed-attempt validation
        try:
            decoded = parse_json_bytes(
                response["text"].encode("utf-8"), label="judge response"
            )
        except StrictJsonError:
            expected_parse = {"status": "invalid", "rating": None, "value": None}
        else:
            if (
                isinstance(decoded, dict)
                and set(decoded) == {"rating"}
                and isinstance(decoded["rating"], str)
                and decoded["rating"] in scale
            ):
                rating = decoded["rating"]
                expected_parse = {
                    "status": "ok",
                    "rating": rating,
                    "value": scale[rating],
                }
            else:
                expected_parse = {
                    "status": "invalid",
                    "rating": None,
                    "value": None,
                }
    elif attempts and attempts[-1]["status"] == "refusal":
        expected_parse = {"status": "refusal", "rating": None, "value": None}
    else:
        expected_parse = {"status": "unavailable", "rating": None, "value": None}
    if parsed != expected_parse:
        _fail("parsed judge result does not match deterministic response replay")
    if (trial["status"] == "complete") != (parsed["status"] == "ok"):
        _fail("trial completeness must agree with its parse outcome")


def _check_retained_inspect_usage(
    usage: object,
    attempt_usage: object,
    *,
    completed: bool,
    maximum_input_tokens: int,
    maximum_output_tokens: int,
) -> None:
    if usage != attempt_usage:
        _fail("retained Inspect token usage does not match its attempt")
    if completed and usage is None:
        _fail("completed retained Inspect call requires token usage")
    if usage is None:
        return
    if not isinstance(usage, dict) or set(usage) != {
        "input_tokens",
        "output_tokens",
    }:
        _fail("retained Inspect token usage is invalid")
    for name, maximum in (
        ("input_tokens", maximum_input_tokens),
        ("output_tokens", maximum_output_tokens),
    ):
        count = usage[name]
        if type(count) is not int or not 0 <= count <= maximum:
            _fail(
                f"retained Inspect {name.replace('_', ' ')} exceeds its per-call reservation"
            )


def _check_retained_inspect_event(
    attempt: object, event: object, collection: dict[str, Any]
) -> None:
    if not isinstance(attempt, dict) or not isinstance(event, dict):
        _fail("retained Inspect attempts and events must be objects")
    required = {
        "event",
        "uuid",
        "role",
        "model",
        "input",
        "tools",
        "tool_choice",
        "config",
        "retries",
        "cache",
        "call",
        "output",
        "error",
    }
    if set(event) != required:
        _fail("retained Inspect model event has unsupported fields")
    mapping = attempt.get("source")
    output = event.get("output")
    call = event.get("call")
    request = attempt.get("request")
    if (
        not isinstance(mapping, dict)
        or event.get("event") != "model"
        or event.get("uuid") != mapping.get("model_event_id")
        or event.get("role") != "grader"
        or not isinstance(output, dict)
        or not isinstance(call, dict)
        or not isinstance(request, dict)
        or event.get("model") is None
    ):
        _fail("retained Inspect model event does not match its attempt")
    if set(output) != {
        "model",
        "request_id",
        "finish_reason",
        "usage",
        "completion",
    }:
        _fail("retained Inspect model output has unsupported fields")
    if (
        event.get("cache") is not None
        or type(event.get("retries")) is not int
        or event.get("retries") != 0
    ):
        _fail("retained Inspect model events cannot use cache or SDK retries")
    if event.get("tools") != [] or event.get("tool_choice") != "none":
        _fail("retained Inspect model events cannot use tools")
    try:
        normalized_request = parse_json_bytes(
            request["text"].encode("utf-8"),
            label="retained Inspect normalized request",
        )
    except (KeyError, AttributeError, StrictJsonError) as exc:
        raise JudgeMeasurementContractError(
            "retained Inspect normalized request is invalid"
        ) from exc
    if (
        not isinstance(normalized_request, dict)
        or event.get("model") != normalized_request.get("model")
        or event.get("input") != normalized_request.get("messages")
    ):
        _fail("retained Inspect input differs from its normalized request")
    expected_config = normalized_request.get("config")
    observed_config = event.get("config")
    if not isinstance(expected_config, dict) or set(expected_config) != {
        "temperature",
        "top_p",
        "max_output_tokens",
        "seed",
        "reasoning_effort",
    }:
        _fail("retained Inspect normalized config must be an object")
    temperature_value = expected_config.get("temperature")
    top_p_value = expected_config.get("top_p")
    if (
        isinstance(temperature_value, bool)
        or not isinstance(temperature_value, (str, int, float))
        or isinstance(top_p_value, bool)
        or not isinstance(top_p_value, (str, int, float))
    ):
        _fail("retained Inspect normalized config is invalid")
    try:
        temperature = float(temperature_value)
        top_p = float(top_p_value)
    except (TypeError, ValueError) as exc:
        raise JudgeMeasurementContractError(
            "retained Inspect normalized config is invalid"
        ) from exc
    if observed_config != {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": expected_config.get("max_output_tokens"),
        "seed": expected_config.get("seed"),
        "reasoning_effort": expected_config.get("reasoning_effort"),
        "max_retries": 0,
        "timeout": collection.get("request_timeout_seconds"),
        "attempt_timeout": collection.get("request_timeout_seconds"),
        "max_connections": collection.get("concurrency"),
        "adaptive_connections": False,
        "num_choices": 1,
        "internal_tools": False,
        "parallel_tool_calls": False,
        "reasoning_summary": "none",
        "reasoning_history": "none",
        "cache": False,
        "batch": False,
    }:
        _fail("retained Inspect generation config differs from its request")
    if set(call) != {"request", "response", "error"} or not isinstance(
        call.get("request"), dict
    ):
        _fail("retained Inspect provider call is incomplete")
    _bounded_canonical(call, JUDGE_REQUEST_MAX_BYTES, "retained Inspect provider call")
    _check_retained_provider_secrets(call)
    _check_inspect_provider_projection(
        call=call,
        output=output,
        normalized_request=normalized_request,
        completed=attempt.get("status") == "completed",
        inspect_version=collection.get("inspect_version"),
    )
    if output.get("model") != attempt.get("resolved_model"):
        _fail("retained Inspect resolved model does not match its attempt")
    response = attempt.get("response")
    completed = attempt.get("status") == "completed"
    _check_retained_inspect_usage(
        output.get("usage"),
        attempt.get("usage"),
        completed=completed,
        maximum_input_tokens=collection["input_tokens_per_call"],
        maximum_output_tokens=observed_config["max_tokens"],
    )
    if completed and (
        event.get("error") is not None
        or call.get("error") not in (None, False)
        or not isinstance(call.get("response"), dict)
    ):
        _fail("completed retained Inspect call has an inconsistent outcome")
    if not completed and (
        event.get("error") is None
        or call.get("error") is not True
        or (
            call.get("response") is not None
            and not isinstance(call.get("response"), dict)
        )
    ):
        _fail("failed retained Inspect call has an inconsistent outcome")
    if not completed:
        event_error = event["error"]
        attempt_error = attempt.get("error")
        if (
            not isinstance(event_error, dict)
            or set(event_error) != {"status", "code", "message"}
            or event_error.get("status") != attempt.get("status")
            or not isinstance(attempt_error, dict)
            or {
                "code": event_error.get("code"),
                "message": event_error.get("message"),
            }
            != attempt_error
        ):
            _fail("retained Inspect error differs from its attempt")
    if response is not None:
        completion = output.get("completion")
        try:
            if not isinstance(completion, str):
                raise AttributeError
            decoded_completion = parse_json_bytes(
                completion.encode("utf-8"), label="Inspect model completion"
            )
        except (AttributeError, StrictJsonError):
            decoded_completion = completion
        if canonical_payload(decoded_completion).decode("utf-8") != response.get(
            "text"
        ):
            _fail("retained Inspect completion does not match its attempt")
    if output.get("request_id") != attempt.get("request_id") or output.get(
        "finish_reason"
    ) != attempt.get("finish_reason"):
        _fail("retained Inspect output metadata does not match its attempt")
    if (event.get("error") is None) != completed:
        _fail("retained Inspect error state does not match its attempt")


def _inspect_source_trials(decoded: dict[str, Any]) -> list[dict[str, Any]]:
    if set(decoded) != {"format", "inspect_version", "collection", "records"}:
        _fail("retained Inspect source has an unsupported shape")
    if (
        decoded["format"] != INSPECT_SOURCE_FORMAT
        or decoded["inspect_version"] not in {"0.3.254", "0.3.263"}
        or not isinstance(decoded["records"], list)
    ):
        _fail("retained Inspect source has an unsupported profile")
    trials: list[dict[str, Any]] = []
    collection = decoded["collection"]
    required_collection = {
        "grader",
        "inspect_version",
        "profile",
        "epochs",
        "log_model_api",
        "log_samples",
        "sdk_max_retries",
        "tools",
        "concurrency",
        "requests_per_minute",
        "request_timeout_seconds",
        "max_calls",
        "max_input_tokens",
        "max_output_tokens",
        "max_cost_microusd",
        "input_tokens_per_call",
        "cost_microusd_per_call",
    }
    if not isinstance(collection, dict) or set(collection) != required_collection:
        _fail("retained Inspect collection options must be an object")
    grader = collection["grader"]
    if (
        not isinstance(grader, str)
        or not 0 < len(grader) <= 256
        or any(part in grader for part in (":", "@", "?", "#", "\\"))
        or any(ord(char) < 33 for char in grader)
        or collection["inspect_version"] != decoded["inspect_version"]
        or collection["profile"] != "inspect-text-frozen-answer-v1"
        or collection["epochs"] != 1
        or type(collection["epochs"]) is not int
        or collection["log_model_api"] is not True
        or collection["log_samples"] is not True
        or collection["sdk_max_retries"] != 0
        or type(collection["sdk_max_retries"]) is not int
        or collection["tools"] is not False
    ):
        _fail("retained Inspect collection configuration is unsupported")
    for name, maximum in (
        ("concurrency", 32),
        ("requests_per_minute", 10000),
        ("request_timeout_seconds", 3600),
        ("max_calls", 600000),
        ("max_input_tokens", 10**12),
        ("max_output_tokens", 10**12),
        ("max_cost_microusd", 10**12),
        ("input_tokens_per_call", 1048576),
        ("cost_microusd_per_call", 10**9),
    ):
        value = collection[name]
        if type(value) is not int or not 1 <= value <= maximum:
            _fail(f"retained Inspect {name} is outside its supported range")
    spent_calls = 0
    reserved_output_tokens = 0
    for record in decoded["records"]:
        if not isinstance(record, dict) or set(record) != {"trial", "events"}:
            _fail("retained Inspect records must contain one trial and its events")
        trial = record["trial"]
        events = record["events"]
        if (
            not isinstance(trial, dict)
            or not isinstance(trial.get("attempts"), list)
            or not isinstance(events, list)
            or len(events) != len(trial["attempts"])
        ):
            _fail("retained Inspect event count must match the trial attempts")
        for attempt, event in zip(trial["attempts"], events, strict=True):
            _check_retained_inspect_event(attempt, event, collection)
            spent_calls += 1
            reserved_output_tokens += event["config"]["max_tokens"]
        trials.append(trial)
    if (
        spent_calls > collection["max_calls"]
        or spent_calls * collection["input_tokens_per_call"]
        > collection["max_input_tokens"]
        or reserved_output_tokens > collection["max_output_tokens"]
        or spent_calls * collection["cost_microusd_per_call"]
        > collection["max_cost_microusd"]
    ):
        _fail("retained Inspect calls exceed their declared resource reservations")
    return trials


def _check_retained_provider_secrets(value: object) -> None:
    blocked = {
        "authorization",
        "api-key",
        "api_key",
        "apikey",
        "access_token",
        "secret",
    }
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                folded = key.casefold() if isinstance(key, str) else ""
                if folded in blocked:
                    _fail("retained Inspect provider call contains a credential field")
                if folded in {"headers", "extra_headers", "http_headers"} and (
                    not isinstance(child, dict)
                    or not {item.casefold() for item in child} <= {"x-irid"}
                    or not all(
                        isinstance(item, str) and 0 < len(item) <= 128
                        for item in child.values()
                    )
                ):
                    _fail("retained Inspect provider call contains unsupported headers")
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)


def _provider_completion(response: dict[str, Any]) -> object:
    if response.get("format") == "invarlock/judge-provider-response-v1":
        if set(response) != {
            "format",
            "content",
            "model",
            "id",
            "finish_reason",
            "usage",
        } or not isinstance(response.get("content"), str):
            _fail("retained normalized provider response has an unsupported shape")
        content = response["content"]
        try:
            return parse_json_bytes(
                content.encode("utf-8"), label="provider completion"
            )
        except StrictJsonError:
            return content
    if set(response) == {"rating"}:
        return response
    choices = response.get("choices")
    if (
        isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], dict)
        and isinstance(choices[0].get("message"), dict)
        and isinstance(choices[0]["message"].get("content"), str)
    ):
        message = choices[0]["message"]
        if (
            message.get("role", "assistant") != "assistant"
            or message.get("refusal") not in (None, "")
            or message.get("tool_calls") not in (None, [])
            or message.get("function_call") is not None
        ):
            _fail(
                "retained Inspect provider response must be a tool-free assistant message without refusal"
            )
        content = message["content"]
        try:
            return parse_json_bytes(
                content.encode("utf-8"), label="provider completion"
            )
        except StrictJsonError:
            return content
    raise JudgeMeasurementContractError(
        "retained Inspect provider response uses an unsupported shape"
    )


def _check_native_inspect_controls(
    request: dict[str, Any],
    expected: dict[str, Any],
    *,
    gpt56_projection: bool = False,
) -> None:
    if gpt56_projection and (
        "temperature" in request
        or Decimal(str(expected["temperature"])) != Decimal(1)
        or "max_completion_tokens" not in request
    ):
        _fail("retained Inspect GPT-5.6 controls differ from its approved projection")
    for provider_key in ("temperature", "top_p"):
        if gpt56_projection and provider_key == "temperature":
            continue
        try:
            if type(request[provider_key]) not in (int, float):
                _fail("retained Inspect provider controls must be numeric")
            observed = Decimal(str(request[provider_key]))
            required = Decimal(str(expected[provider_key]))
        except (KeyError, TypeError, ValueError) as exc:
            raise JudgeMeasurementContractError(
                "retained Inspect provider config is incomplete"
            ) from exc
        if observed != required:
            _fail("retained Inspect provider config differs from the request")
    seed = request.get("seed")
    if seed != expected["seed"] or (seed is not None and type(seed) is not int):
        _fail("retained Inspect provider seed differs from the request")
    if len({"max_tokens", "max_completion_tokens"} & request.keys()) != 1:
        _fail("retained Inspect provider request must use one token limit")
    token_limit = request.get("max_tokens", request.get("max_completion_tokens"))
    if type(token_limit) is not int or token_limit != expected["max_output_tokens"]:
        _fail("retained Inspect provider token limit differs from the request")
    if type(request.get("n", 1)) is not int or request.get("n", 1) != 1:
        _fail("retained Inspect provider request must select one completion")
    reasoning_effort = expected["reasoning_effort"]
    if reasoning_effort is None:
        if request.get("reasoning_effort") is not None:
            _fail("retained Inspect provider reasoning effort differs from the request")
    elif request.get("reasoning_effort") != reasoning_effort:
        _fail("retained Inspect provider reasoning effort differs from the request")


def _check_inspect_provider_response(
    *, response: object, output: dict[str, Any], request: dict[str, Any]
) -> None:
    if not isinstance(response, dict):
        _fail("completed retained Inspect provider response must be an object")
    normalized_response = (
        response.get("format") == "invarlock/judge-provider-response-v1"
    )
    if "service_tier" in request and response.get("service_tier") != "default":
        _fail("retained Inspect provider response contradicts its service tier")
    provider_completion = _provider_completion(response)
    completion = output.get("completion")
    if not isinstance(completion, str):
        _fail("retained Inspect output completion must be text")
    try:
        projected_completion: object = parse_json_bytes(
            completion.encode("utf-8"), label="Inspect output completion"
        )
    except StrictJsonError:
        projected_completion = completion
    if canonical_payload(provider_completion) != canonical_payload(
        projected_completion
    ):
        _fail("retained Inspect provider response contradicts its completion")
    if isinstance(response.get("model"), str) and response["model"] != output.get(
        "model"
    ):
        _fail("retained Inspect provider response contradicts its resolved model")
    if isinstance(response.get("id"), str) and response["id"] != output.get(
        "request_id"
    ):
        _fail("retained Inspect provider response contradicts its request ID")
    if normalized_response:
        if response.get("model") != output.get("model"):
            _fail("retained Inspect provider response contradicts its resolved model")
        if response.get("id") != output.get("request_id"):
            _fail("retained Inspect provider response contradicts its request ID")
        if response.get("finish_reason") != output.get("finish_reason"):
            _fail("retained Inspect provider response contradicts its finish reason")
        if response.get("usage") != output.get("usage"):
            _fail("retained Inspect provider usage contradicts its output")
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        # Chat Completions and Inspect use different names for token exhaustion.
        # Preserve the native value while checking the pinned interpretation;
        # tool-call endings remain unsupported by this tool-free profile.
        stop_reasons = {
            "stop": "stop",
            "length": "max_tokens",
            "content_filter": "content_filter",
        }
        finish_reason = choices[0].get("finish_reason")
        if (
            not isinstance(finish_reason, str)
            or finish_reason not in stop_reasons
            or stop_reasons[finish_reason] != output.get("finish_reason")
        ):
            _fail("retained Inspect provider response contradicts its finish reason")
    usage = response.get("usage")
    projected_usage = output.get("usage")
    if isinstance(usage, dict) and isinstance(projected_usage, dict):
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if input_tokens != projected_usage.get(
            "input_tokens"
        ) or output_tokens != projected_usage.get("output_tokens"):
            _fail("retained Inspect provider usage contradicts its output")


def _check_inspect_provider_projection(
    *,
    call: dict[str, Any],
    output: dict[str, Any],
    normalized_request: dict[str, Any],
    completed: bool,
    inspect_version: object,
) -> None:
    request = call["request"]
    normalized = request.get("format") == "invarlock/judge-request-v1"
    # This is the qualified, version-bound SDK projection, not permission for
    # arbitrary role changes or missing sampling controls on other models.
    gpt56_projection = (
        not normalized
        and inspect_version == "0.3.263"
        and normalized_request["model"] in _INSPECT_0_3_263_REASONING_EFFORTS
    )
    if normalized:
        if canonical_payload(request) != canonical_payload(normalized_request):
            _fail(
                "retained Inspect normalized provider request differs from the request"
            )
    else:
        allowed_controls = {
            "model",
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "seed",
            "reasoning_effort",
            "n",
            "tools",
            "tool_choice",
            "extra_headers",
            "service_tier",
        }
        if set(request) - allowed_controls:
            _fail("retained Inspect provider request contains unsupported controls")
        # Historical requests omitted this field. Preserve their exact bytes and
        # meaning; explicit new requests may select only standard processing.
        if "service_tier" in request and request["service_tier"] != "default":
            _fail("retained Inspect provider request has an unsupported service tier")
    messages = request.get("messages")
    if not isinstance(messages, list):
        _fail("retained Inspect provider request must contain chat messages")
    provider_messages: list[dict[str, str]] = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
        ):
            _fail("retained Inspect provider messages must be plain text")
        provider_messages.append(
            {"role": message["role"], "content": message["content"]}
        )
    expected_messages = normalized_request["messages"]
    if gpt56_projection:
        expected_messages = [
            {**message, "role": "developer"} if message["role"] == "system" else message
            for message in expected_messages
        ]
    if provider_messages != expected_messages:
        _fail("retained Inspect provider messages differ from the approved request")
    requested_model = normalized_request["model"]
    provider_model = request.get("model")
    if not isinstance(provider_model, str) or provider_model not in {
        requested_model,
        requested_model.split("/", 1)[-1],
    }:
        _fail("retained Inspect provider model differs from the approved request")
    expected = normalized_request["config"]
    if not normalized:
        _check_native_inspect_controls(
            request, expected, gpt56_projection=gpt56_projection
        )
    # Inspect 0.3.254 retains OpenAI's NOT_GIVEN values as JSON null. Both
    # absent/null and an empty list describe the same tool-free request.
    if request.get("tools") not in (None, []):
        _fail("retained Inspect provider request contains tools")
    if request.get("tool_choice") not in (None, "none"):
        _fail("retained Inspect provider request contains an unsupported tool choice")
    if not completed:
        return
    _check_inspect_provider_response(
        response=call["response"], output=output, request=request
    )


def _source_trials(
    source: dict[str, Any],
    *,
    inspect_collections: list[dict[str, Any]],
    runtime_sources: list[dict[str, Any]],
    openai_compatible_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    content = source["content"]
    encoded = content.encode("utf-8")
    if len(encoded) != source["byte_size"]:
        _fail(f"source {source['source_id']!r} byte_size does not match content")
    if _sha256(encoded) != source["sha256"]:
        _fail(f"source {source['source_id']!r} digest does not match content")
    try:
        decoded = parse_json_bytes(encoded, label=f"source {source['source_id']!r}")
    except StrictJsonError as exc:
        raise JudgeMeasurementContractError(str(exc)) from exc
    if canonical_payload(decoded) != encoded:
        _fail(f"source {source['source_id']!r} content must use canonical JSON")
    if not isinstance(decoded, dict):
        _fail(f"source {source['source_id']!r} has an unsupported retained shape")
    if source["profile"] == "retained-inspect-model-events-v1":
        trials = _inspect_source_trials(decoded)
        inspect_collections.append(decoded["collection"])
        return trials
    if source["profile"] == RUNTIME_PROVIDER_SOURCE_FORMAT:
        required = {
            "format",
            "runtime_spec",
            "artifact_identity",
            "scoring_observation",
            "provider_receipt",
            "trials",
        }
        if (
            set(decoded) != required
            or decoded.get("format") != RUNTIME_PROVIDER_SOURCE_FORMAT
            or not isinstance(decoded.get("trials"), list)
        ):
            _fail(
                f"source {source['source_id']!r} has an unsupported runtime-provider profile"
            )
        runtime_sources.append(decoded)
        trials = cast(list[Any], decoded["trials"])
        if any(
            not isinstance(trial, dict)
            or not isinstance(trial.get("attempts"), list)
            or any(not isinstance(attempt, dict) for attempt in trial["attempts"])
            for trial in trials
        ):
            _fail("retained runtime-provider trials must contain attempt objects")
        return trials
    if source["profile"] == OPENAI_COMPATIBLE_SOURCE_PROFILE:
        required = {
            "format",
            "collection",
            "http",
            "service_identity",
            "trials",
        }
        if (
            set(decoded) != required
            or decoded.get("format") != OPENAI_COMPATIBLE_SOURCE_FORMAT
            or not isinstance(decoded.get("trials"), list)
        ):
            _fail(
                f"source {source['source_id']!r} has an unsupported OpenAI-compatible profile"
            )
        openai_compatible_sources.append(decoded)
        trials = cast(list[Any], decoded["trials"])
        if any(
            not isinstance(trial, dict)
            or not isinstance(trial.get("attempts"), list)
            or any(not isinstance(attempt, dict) for attempt in trial["attempts"])
            for trial in trials
        ):
            _fail("retained OpenAI-compatible trials must contain attempt objects")
        return trials
    if (
        set(decoded) != {"format", "trials"}
        or decoded.get("format") != SOURCE_FORMAT
        or not isinstance(decoded.get("trials"), list)
    ):
        _fail(f"source {source['source_id']!r} has an unsupported retained profile")
    for trial in decoded["trials"]:
        if not isinstance(trial, dict) or not isinstance(trial.get("attempts"), list):
            _fail("retained source trials must be objects with attempt arrays")
        if any(not isinstance(attempt, dict) for attempt in trial["attempts"]):
            _fail("retained source attempts must be objects")
    return cast(list[dict[str, Any]], decoded["trials"])


def _runtime_provider_source_errors(
    source: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    """Replay one local runtime source without importing its provider."""

    errors: list[str] = []
    raw_spec = source.get("runtime_spec")
    if not isinstance(raw_spec, dict) or set(raw_spec) != {
        "provider_name",
        "model_id",
        "settings",
    }:
        return ["retained runtime judge spec has an unsupported shape"]
    try:
        spec = ModelRuntimeSpec(
            provider_name=raw_spec["provider_name"],
            model_id=raw_spec["model_id"],
            settings=raw_spec["settings"],
        )
        artifact_bytes = canonical_payload(source["artifact_identity"])
        observation_bytes = canonical_payload(source["scoring_observation"])
        receipt_bytes = canonical_payload(source["provider_receipt"])
        artifact = decode_artifact_identity(artifact_bytes)
        observation = decode_scoring_observation(observation_bytes)
        receipt = decode_runtime_provider_receipt(receipt_bytes)
    except (KeyError, TypeError, ValueError, RuntimeProviderEvidenceError) as exc:
        return [f"retained runtime judge evidence is invalid: {str(exc)[:240]}"]

    judge = plan["judge"]
    if spec.provider_name not in {"hf_transformers", "llama_cpp"}:
        errors.append("retained runtime judge provider is unsupported")
    if spec.provider_name != judge["provider"]:
        errors.append("retained runtime judge provider differs from the plan")
    if spec.model_id != judge["requested_model"] or judge[
        "approved_resolved_models"
    ] != [spec.model_id]:
        errors.append("retained runtime judge model differs from the plan")
    identity = judge["model_identity"]
    if identity.get("kind") != "local_weights" or identity.get(
        "weights_sha256"
    ) != artifact_identity_sha256(artifact):
        errors.append("retained runtime judge artifact identity differs from the plan")
    if receipt.plugin.distribution != "invarlock":
        errors.append("retained runtime judge provider is not first-party")
    if receipt.outer_image_digest is None:
        errors.append("retained runtime judge lacks a pinned outer image")
    errors.extend(
        runtime_provider_evidence_errors(
            artifact_identity=artifact,
            scoring_observation=observation,
            receipt=receipt,
            scoring_observation_bytes=observation_bytes,
            expected_outer_image_digest=receipt.outer_image_digest,
        )
    )
    errors.extend(
        runtime_request_binding_errors(
            provider_name=spec.provider_name,
            settings=spec.settings,
            artifact_identity=artifact,
            receipt=receipt,
        )
    )
    if (
        "text_causal" not in receipt.capabilities.tasks
        or "exact_match" not in receipt.capabilities.metrics
    ):
        errors.append("retained runtime judge lacks required text capabilities")
    execution = receipt.execution_settings
    config = judge["config"]
    settings = spec.settings
    context_length = settings.get("context_length")
    max_output_tokens = settings.get("max_output_tokens")
    timeout_seconds = settings.get("timeout_seconds")
    if (
        execution.allow_network
        or execution.seed != config["seed"]
        or execution.max_output_tokens != config["max_output_tokens"]
        or config["temperature"] != "0"
        or config["top_p"] != "1"
        or config["reasoning_effort"] is not None
    ):
        errors.append("retained runtime judge generation settings differ from the plan")
    if (
        type(settings.get("seed")) is not int
        or settings.get("seed") != config["seed"]
        or type(max_output_tokens) is not int
        or max_output_tokens != config["max_output_tokens"]
        or type(settings.get("batch_size")) is not int
        or settings.get("batch_size") != 1
        or type(context_length) is not int
        or not 1 <= context_length <= 1024 * 1024
        or max_output_tokens > context_length
        or type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= 604800
    ):
        errors.append("retained runtime judge spec violates strict execution bounds")
    plan_digest = _sha256(canonical_payload(plan))
    if observation.schedule_sha256 != plan_digest:
        errors.append("retained runtime judge observation differs from the plan")
    aggregate = runtime_scoring_records_sha256(
        [cast(dict[str, object], asdict(record)) for record in observation.records]
    )
    if observation.aggregate_source_sha256 != aggregate:
        errors.append("retained runtime judge observation aggregate is invalid")

    trials = cast(list[dict[str, Any]], source["trials"])
    if len(observation.records) != len(trials):
        errors.append("retained runtime judge observation coverage is incomplete")
        return errors
    for index, (record, trial) in enumerate(
        zip(observation.records, trials, strict=True)
    ):
        attempts = trial.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 1:
            errors.append("retained runtime judge trial must contain one attempt")
            continue
        attempt = attempts[0]
        request = attempt.get("request")
        if not isinstance(request, dict) or not isinstance(request.get("text"), str):
            errors.append("retained runtime judge request is invalid")
            continue
        prompt = request["text"]
        try:
            runtime_prompt = render_runtime_prompt(
                cast(JudgeMeasurementPlan, plan), prompt.encode("utf-8")
            )
        except (UnicodeError, ValueError) as exc:
            errors.append(f"retained runtime judge prompt is invalid: {str(exc)[:240]}")
            continue
        part = EvaluationInputPart(
            kind="text",
            role="prompt",
            text=runtime_prompt.decode("utf-8"),
            sha256=_sha256(runtime_prompt),
        )
        if (
            record.record_id != trial.get("trial_id")
            or record.input_sha256 != evaluation_input_parts_sha256((part,))
            or attempt.get("source", {}).get("record_index") != index
            or attempt.get("source", {}).get("model_event_id") != record.record_id
        ):
            errors.append("retained runtime judge record does not match its trial")
        if record.status == "ok":
            response = attempt.get("response")
            if (
                attempt.get("status") != "completed"
                or record.output_text is None
                or not isinstance(response, dict)
                or response.get("text") != record.output_text
                or response.get("sha256") != record.output_sha256
            ):
                errors.append("retained runtime judge output differs from its attempt")
        elif (
            attempt.get("status") != "cancelled"
            or attempt.get("response") is not None
            or not isinstance(attempt.get("error"), dict)
        ):
            errors.append("retained runtime judge error differs from its attempt")
    return errors


def _check_runtime_provider_sources(
    runtime_sources: list[dict[str, Any]], plan: dict[str, Any]
) -> None:
    if not runtime_sources:
        return
    judge = plan["judge"]
    config = judge["config"]
    schedule = plan["schedule"]
    if (
        judge["model_identity"]["kind"] != "local_weights"
        or judge["provider"] not in {"hf_transformers", "llama_cpp"}
        or judge["approved_resolved_models"] != [judge["requested_model"]]
        or config["temperature"] != "0"
        or config["top_p"] != "1"
        or config["seed"] is None
        or config["reasoning_effort"] is not None
        or schedule["max_attempts"] != 1
        or schedule["retry_on"]
        or schedule["cache"] != "forbid"
    ):
        _fail("runtime judge plan does not satisfy the strict local profile")
    expected_spec = runtime_sources[0]["runtime_spec"]
    expected_artifact = runtime_sources[0]["artifact_identity"]
    expected_receipt = dict(runtime_sources[0]["provider_receipt"])
    expected_receipt.pop("scoring_observation_sha256", None)
    for source in runtime_sources:
        receipt = dict(source["provider_receipt"])
        receipt.pop("scoring_observation_sha256", None)
        if (
            source["runtime_spec"] != expected_spec
            or source["artifact_identity"] != expected_artifact
            or receipt != expected_receipt
        ):
            _fail("retained runtime judge shards have inconsistent runtime identity")
        errors = _runtime_provider_source_errors(source, plan)
        if errors:
            _fail(errors[0])


def _compatible_credential_field(value: Any) -> bool:
    return has_credential_field(value)


def _openai_compatible_source_errors(
    source: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    """Replay one compatible-service source without contacting its endpoint."""

    errors: list[str] = []
    collection = source.get("collection")
    http = source.get("http")
    identity = source.get("service_identity")
    trials = source.get("trials")
    if (
        not isinstance(collection, dict)
        or not isinstance(http, dict)
        or set(http)
        != {
            "outcome",
            "request",
            "response_status",
            "response_headers",
            "response_body",
        }
        or not isinstance(identity, dict)
        or set(identity)
        != {
            "service",
            "endpoint_sha256",
            "requested_model",
            "response_model",
            "request_id",
            "system_fingerprint",
        }
        or not isinstance(trials, list)
        or len(trials) != 1
        or not isinstance(trials[0], dict)
    ):
        return ["retained OpenAI-compatible source has an unsupported shape"]
    try:
        checked = normalize_configuration(
            collection, maximum_input_bytes=MEASUREMENTS_MAX_BYTES
        )
    except OpenAICompatibleContractError:
        return ["retained compatible endpoint configuration is invalid"]
    if checked != collection:
        errors.append("retained compatible endpoint is not canonical")
    judge = plan["judge"]
    config = judge["config"]
    schedule = plan["schedule"]
    expected_identity = service_identity(checked)
    if (
        identity["service"] != expected_identity["service"]
        or identity["endpoint_sha256"] != expected_identity["endpoint_sha256"]
        or identity["requested_model"] != collection["model"]
    ):
        errors.append("retained compatible endpoint identity is invalid")
    if (
        judge["provider"] != "openai_compatible"
        or judge["requested_model"] != collection["model"]
        or collection["model"] not in judge["approved_resolved_models"]
        or judge["model_identity"] != {"kind": "hosted_api", "weights_sha256": None}
        or judge.get("service_identity") != expected_identity
        or config["reasoning_effort"] is not None
        or config["seed"] is None
        or schedule["max_attempts"] != 1
        or schedule["retry_on"]
        or schedule["cache"] != "forbid"
    ):
        errors.append("retained compatible source differs from the strict plan")
    trial = cast(dict[str, Any], trials[0])
    attempts = trial.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        errors.append("retained compatible trial must contain one attempt")
        return errors
    attempt = attempts[0]
    request_blob = attempt.get("request")
    response_body = http.get("response_body")
    if not isinstance(request_blob, dict) or not isinstance(
        request_blob.get("text"), str
    ):
        errors.append("retained compatible request material is invalid")
        return errors
    try:
        normalized = parse_json_bytes(
            request_blob["text"].encode("utf-8"), label="retained compatible request"
        )
    except (StrictJsonError, UnicodeEncodeError) as exc:
        return [f"retained compatible request material is invalid: {str(exc)[:240]}"]
    if not isinstance(normalized, dict):
        return ["retained compatible normalized request must be an object"]
    expected_wire = wire_request(
        response_format_profile=collection.get("response_format", "json_object"),
        model=collection["model"],
        messages=normalized.get("messages"),
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_output_tokens"],
        seed=config["seed"],
        rating_labels=[item["label"] for item in plan["scale"]["ratings"]],
    )
    try:
        wire_matches = canonical_payload(http["request"]) == canonical_payload(
            expected_wire
        )
    except (TypeError, ValueError):
        wire_matches = False
    if not wire_matches or _compatible_credential_field(http["request"]):
        errors.append(
            "retained compatible wire request differs from the approved request"
        )

    headers = http.get("response_headers")
    if not isinstance(headers, dict) or any(
        name not in {"content-type", "server", "x-request-id"}
        or not isinstance(value, str)
        or len(value) > 512
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159 for character in value
        )
        for name, value in headers.items()
    ):
        errors.append("retained compatible response headers are invalid")
    status = http.get("response_status")
    outcome = http.get("outcome")
    allowed_outcomes = {
        "success",
        "http_error",
        "transport_error",
        "malformed_response",
        "response_too_large",
        "credential_echo",
    }
    if not isinstance(outcome, str) or outcome not in allowed_outcomes:
        errors.append("retained compatible HTTP outcome is invalid")
        return errors
    if outcome == "transport_error":
        valid_status = status is None
    else:
        valid_status = type(status) is int and 100 <= status <= 599
    if not valid_status:
        errors.append("retained compatible HTTP status is invalid")
        return errors

    raw_response: bytes | None = None
    decoded_response: Any = None
    if response_body is not None:
        try:
            raw_response = decode_http_blob(response_body)
        except OpenAICompatibleContractError:
            errors.append("retained compatible response body is invalid")
            return errors
        if len(raw_response) > 2 * 1024 * 1024:
            errors.append("retained compatible response body exceeds its byte limit")
            return errors
        try:
            decoded_response = parse_json_bytes(
                raw_response, label="retained compatible response"
            )
        except StrictJsonError:
            decoded_response = None
        if _compatible_credential_field(decoded_response):
            errors.append("retained compatible response contains credential material")

    if outcome == "success":
        if status != 200 or raw_response is None:
            errors.append("retained compatible success response is incomplete")
            return errors
        try:
            facts = response_facts(
                decoded_response,
                approved_models=judge["approved_resolved_models"],
                max_output_tokens=judge["config"]["max_output_tokens"],
            )
        except OpenAICompatibleContractError:
            errors.append("retained compatible success response is invalid")
            return errors
        content = facts["content"]
        response = attempt.get("response")
        if (
            attempt.get("status") != "completed"
            or not isinstance(response, dict)
            or response.get("text") != content
            or response.get("sha256") != _text_sha256(content)
            or attempt.get("resolved_model") != facts["model"]
            or attempt.get("request_id") != facts["request_id"]
            or attempt.get("finish_reason") != facts["finish_reason"]
            or attempt.get("usage") != facts["usage"]
            or identity["response_model"] != facts["model"]
            or identity["request_id"] != facts["request_id"]
            or identity["system_fingerprint"] != facts["system_fingerprint"]
        ):
            errors.append("retained compatible completion identity is invalid")
    else:
        if outcome == "http_error" and (
            status == 200
            or (response_body is None and not 300 <= cast(int, status) <= 399)
        ):
            errors.append("retained compatible HTTP error outcome is invalid")
        if outcome in {"transport_error", "response_too_large", "credential_echo"} and (
            response_body is not None
        ):
            errors.append("retained compatible unsafe failure retained a response body")
        if outcome in {"transport_error", "credential_echo"} and headers:
            errors.append(
                "retained compatible unsafe failure retained response headers"
            )
        expected_code, expected_message = failure_details(outcome, status)
        error = attempt.get("error")
        if (
            attempt.get("status") != "cancelled"
            or attempt.get("response") is not None
            or not isinstance(error, dict)
            or error.get("code") != expected_code
            or error.get("message") != expected_message
            or attempt.get("resolved_model") is not None
            or attempt.get("request_id") is not None
            or attempt.get("finish_reason") is not None
            or attempt.get("usage") is not None
            or identity["response_model"] is not None
            or identity["request_id"] is not None
            or identity["system_fingerprint"] is not None
        ):
            errors.append("retained compatible failed outcome is invalid")
    return errors


def _check_openai_compatible_sources(
    sources: list[dict[str, Any]], plan: dict[str, Any]
) -> None:
    if not sources:
        return
    collection = sources[0].get("collection")
    spent_input_bytes = 0
    for source in sources:
        if source.get("collection") != collection:
            _fail("retained compatible shards have inconsistent endpoint configuration")
        errors = _openai_compatible_source_errors(source, plan)
        if errors:
            _fail(errors[0])
        spent_input_bytes += len(canonical_payload(source["http"]["request"]))
    assert isinstance(collection, dict)
    spent_calls = len(sources)
    if (
        spent_calls > collection["max_calls"]
        or spent_input_bytes > collection["max_input_bytes"]
        or spent_calls * plan["judge"]["config"]["max_output_tokens"]
        > collection["max_output_tokens"]
    ):
        _fail("retained compatible shards exceed their aggregate resource reservations")


def _frozen_run_records(
    plan: dict[str, Any],
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    try:
        for side, run in (("baseline", baseline_run), ("subject", subject_run)):
            _check_run(run)
            if run_digest(run) != plan[f"{side}_run_sha256"]:
                _fail(f"{side} run does not match the approved plan digest")
            validate_run_case_set(run, plan["case_set_sha256"])
    except JudgeMeasurementContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise JudgeMeasurementContractError(
            f"frozen answer run is invalid: {str(exc)[:240]}"
        ) from exc
    records = {
        "baseline": {row["id"]: row for row in baseline_run["records"]},
        "subject": {row["id"]: row for row in subject_run["records"]},
    }
    planned = {item["case_id"] for item in plan["answer_bindings"]}
    if any(set(side_records) != planned for side_records in records.values()):
        _fail("frozen run membership must exactly match the judging plan")
    from invarlock.evaluator_capture import verify_input_pair

    for case_id in planned:
        verify_input_pair(records["baseline"][case_id], records["subject"][case_id])
    return records


def _validate_frozen_answer_bindings(
    plan: JudgeMeasurementPlan,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
) -> None:
    plan_raw = cast(dict[str, Any], plan)
    records = _frozen_run_records(plan_raw, baseline_run, subject_run)
    for binding in plan_raw["answer_bindings"]:
        case_id = binding["case_id"]
        for side in ("baseline", "subject"):
            record = records[side][case_id]
            if record["error"] is not None:
                _fail(f"case {case_id!r} cannot grade a failed frozen answer")
            if not isinstance(record["input"], str) or not isinstance(
                record["output"], str
            ):
                _fail(
                    "the text-frozen-answer profile requires string inputs and answers"
                )
            answer_key = f"{side}_answer_sha256"
            if binding[answer_key] != _text_sha256(record["output"]):
                _fail(f"case {case_id!r} does not bind the frozen {side} answer")
            request = render_judge_request(
                plan,
                input_text=record["input"],
                answer_text=record["output"],
                reference_text=record["expected"],
            )
            request_key = f"{side}_request_sha256"
            if binding[request_key] != _sha256(request):
                _fail(f"case {case_id!r} request binding does not match frozen inputs")
    return None


def _require_declared_source_profile(raw: dict[str, Any]) -> None:
    if any(source["profile"] != raw["source_profile"] for source in raw["sources"]):
        _fail("measurement source profile differs from the declared profile")


def _check_inspect_shard_budgets(
    inspect_collections: list[dict[str, Any]],
    replayed: dict[str, dict[str, Any]],
    plan_raw: dict[str, Any],
) -> None:
    if inspect_collections:
        collection = inspect_collections[0]
        if any(item != collection for item in inspect_collections[1:]):
            _fail("retained Inspect shards must bind the same collection options")
        _validate_inspect_plan_collection_identity(
            cast(JudgeMeasurementPlan, plan_raw),
            grader=collection["grader"],
            inspect_version=collection["inspect_version"],
        )
        spent_calls = sum(len(trial["attempts"]) for trial in replayed.values())
        if (
            spent_calls > collection["max_calls"]
            or spent_calls * collection["input_tokens_per_call"]
            > collection["max_input_tokens"]
            or spent_calls * plan_raw["judge"]["config"]["max_output_tokens"]
            > collection["max_output_tokens"]
            or spent_calls * collection["cost_microusd_per_call"]
            > collection["max_cost_microusd"]
        ):
            _fail(
                "retained Inspect shards exceed their aggregate resource reservations"
            )


def validate_measurements(
    value: JudgeMeasurements,
    plan: JudgeMeasurementPlan,
    *,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
) -> None:
    """Replay retained source normalization and validate every planned slot."""

    validate_measurement_plan(plan)
    raw = cast(dict[str, Any], value)
    plan_raw = cast(dict[str, Any], plan)
    _precheck_measurement_counts(raw)
    _bounded_canonical(raw, MEASUREMENTS_MAX_BYTES, "judge measurements")
    _validate_schema(raw, "measurements")
    for field in ("expected_trials", "recorded_trials", "completed_trials"):
        _require_integer(raw["completeness"][field], f"completeness {field}")
    for source in raw["sources"]:
        _require_integer(source["byte_size"], "source byte_size")
    for trial in raw["trials"]:
        _check_trial_integer_types(trial)
    _require_declared_source_profile(raw)

    plan_sha256 = _sha256(canonical_payload(plan_raw))
    if raw["plan_sha256"] != plan_sha256:
        _fail("measurements do not bind the supplied plan")

    _validate_frozen_answer_bindings(plan, baseline_run, subject_run)

    sources: dict[str, dict[str, Any]] = {}
    replayed: dict[str, dict[str, Any]] = {}
    inspect_collections: list[dict[str, Any]] = []
    runtime_sources: list[dict[str, Any]] = []
    openai_compatible_sources: list[dict[str, Any]] = []
    for source in raw["sources"]:
        source_id = source["source_id"]
        if source_id in sources:
            _fail("measurement source IDs must be unique")
        sources[source_id] = source
        for record_index, trial in enumerate(
            _source_trials(
                source,
                inspect_collections=inspect_collections,
                runtime_sources=runtime_sources,
                openai_compatible_sources=openai_compatible_sources,
            )
        ):
            _check_trial_integer_types(trial)
            trial_id = trial.get("trial_id")
            if not isinstance(trial_id, str) or trial_id in replayed:
                _fail("retained sources must contain unique trial IDs")
            for attempt_index, attempt in enumerate(trial.get("attempts", [])):
                mapping = attempt.get("source")
                if not isinstance(mapping, dict) or (
                    mapping.get("source_id") != source_id
                    or mapping.get("record_index") != record_index
                    or mapping.get("attempt_index") != attempt_index
                ):
                    _fail("retained source position mapping is inconsistent")
            replayed[trial_id] = trial

    _check_inspect_shard_budgets(inspect_collections, replayed, plan_raw)
    _check_runtime_provider_sources(runtime_sources, plan_raw)
    _check_openai_compatible_sources(openai_compatible_sources, plan_raw)

    bindings = {item["case_id"]: item for item in plan_raw["answer_bindings"]}
    repetitions = plan_raw["schedule"]["repetitions"]
    expected_slots = {
        (case_id, side, repetition)
        for case_id in bindings
        for side in ("baseline", "subject")
        for repetition in range(1, repetitions + 1)
    }
    seen_slots: set[tuple[str, str, int]] = set()
    seen_ids: set[str] = set()
    completed = 0
    source_positions: set[tuple[str, int, int]] = set()
    source_events: set[tuple[str, str]] = set()
    provider_response_ids: set[str] = set()

    for trial in raw["trials"]:
        trial_raw = cast(dict[str, Any], trial)
        slot = (trial_raw["case_id"], trial_raw["side"], trial_raw["repetition"])
        if slot not in expected_slots or slot in seen_slots:
            _fail("measurements contain an unknown or duplicate planned trial slot")
        seen_slots.add(slot)
        expected_id = expected_trial_id(plan_sha256, *slot)
        if trial_raw["trial_id"] != expected_id or expected_id in seen_ids:
            _fail("measurement trial ID does not match its planned slot")
        seen_ids.add(expected_id)
        if trial_raw["plan_sha256"] != plan_sha256:
            _fail(f"trial {expected_id!r} does not bind the supplied plan")
        binding = bindings[trial_raw["case_id"]]
        answer_key = f"{trial_raw['side']}_answer_sha256"
        if trial_raw["answer_sha256"] != binding[answer_key]:
            _fail(f"trial {expected_id!r} does not bind the frozen answer")
        request_key = f"{trial_raw['side']}_request_sha256"
        _check_attempts(
            trial_raw,
            plan=plan_raw,
            source_ids=set(sources),
            expected_request_sha256=binding[request_key],
        )
        for attempt in trial_raw["attempts"]:
            position = (
                attempt["source"]["source_id"],
                attempt["source"]["record_index"],
                attempt["source"]["attempt_index"],
            )
            if position in source_positions:
                _fail("source record positions must be unique across trials")
            source_positions.add(position)
            event = (
                raw["source_profile"]
                if inspect_collections
                else attempt["source"]["source_id"],
                attempt["source"]["model_event_id"],
            )
            if event in source_events:
                _fail("retained model events must belong to exactly one attempt")
            source_events.add(event)
            # The pinned Inspect projection records the provider response ID.
            # It cannot represent independent calls in separate retained shards.
            # Generic retained JSON does not promise this identity semantics.
            response_id = attempt["request_id"]
            if (
                inspect_collections or openai_compatible_sources
            ) and response_id is not None:
                if response_id in provider_response_ids:
                    _fail("retained provider response IDs must be unique")
                provider_response_ids.add(response_id)
        if trial_raw["status"] == "complete":
            completed += 1
        if replayed.get(expected_id) != trial_raw:
            _fail(f"retained source replay differs for trial {expected_id!r}")

    if set(replayed) != seen_ids:
        _fail("retained source replay contains omitted or extra trials")
    if seen_slots != expected_slots:
        _fail("measurements omit one or more planned trial slots")

    completeness = raw["completeness"]
    expected_count = len(expected_slots)
    if completeness["expected_trials"] != expected_count:
        _fail("measurement expected-trial count is inconsistent")
    if completeness["recorded_trials"] != len(raw["trials"]):
        _fail("measurement recorded-trial count is inconsistent")
    if completeness["completed_trials"] != completed:
        _fail("measurement completed-trial count is inconsistent")
    expected_status = "complete" if completed == expected_count else "incomplete"
    if completeness["status"] != expected_status:
        _fail("measurement completeness status is inconsistent")


def load_measurements(
    path: Path,
    *,
    plan: JudgeMeasurementPlan,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
) -> JudgeMeasurements:
    """Read and validate retained measurements from one bounded snapshot."""

    decoded = _load_object(
        Path(path), maximum=MEASUREMENTS_MAX_BYTES, label="judge measurements"
    )
    measurements = cast(JudgeMeasurements, decoded)
    validate_measurements(
        measurements,
        plan,
        baseline_run=baseline_run,
        subject_run=subject_run,
    )
    return measurements


__all__ = [
    "JudgeMeasurementContractError",
    "JUDGE_REQUEST_MAX_BYTES",
    "MEASUREMENTS_FORMAT",
    "MEASUREMENTS_MAX_BYTES",
    "PLAN_FORMAT",
    "PLAN_MAX_BYTES",
    "SOURCE_FORMAT",
    "TRIAL_ID_SCHEME",
    "canonical_payload",
    "expected_trial_id",
    "load_measurement_plan",
    "load_measurements",
    "measurement_plan_digest",
    "render_judge_request",
    "render_runtime_prompt",
    "validate_measurement_plan",
    "validate_measurements",
]
