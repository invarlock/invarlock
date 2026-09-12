"""Strict collection inputs and pure replay of expanded Inspect model events."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
from dataclasses import asdict, dataclass, fields
from decimal import Decimal
from typing import Any, cast

from invarlock.evidence_pack_json import StrictJsonError, parse_json_bytes
from invarlock.judge_measurement_types import (
    JudgeMeasurementPlan,
    JudgeMeasurements,
)
from invarlock.judge_measurements.contracts import (
    JUDGE_REQUEST_MAX_BYTES,
    MEASUREMENTS_MAX_BYTES,
    canonical_payload,
    expected_trial_id,
    measurement_plan_digest,
    render_judge_request,
    validate_measurements,
)

INSPECT_VERSION = "0.3.263"
REPLAY_INSPECT_VERSIONS = {"0.3.254", INSPECT_VERSION}
EXPORT_FORMAT = "invarlock/inspect-judge-export-v1"
EXPORT_PROFILE = "inspect-text-frozen-answer-v1"
RETAINED_SOURCE_PROFILE = "retained-inspect-model-events-v1"
RETAINED_SOURCE_FORMAT = "invarlock/retained-inspect-model-events-v1"
MAX_EXPORT_BYTES = MEASUREMENTS_MAX_BYTES
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_SOURCES = 1000
MAX_RETAINED_EVENT_BYTES = 2 * 1024 * 1024
# A retained event is at most 2 MiB. Each normalized request/response is at
# most 1 MiB and can double when JSON-quoted; the source is quoted once more
# in measurements, which also repeat the trial. 20 MiB covers this growth
# plus fixed metadata. Reserve remapping overhead separately for all slots.
MAX_ADMISSION_GROWTH_BYTES = 20 * 1024 * 1024


class InspectJudgeError(ValueError):
    """Collection configuration or exported material is unsupported."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InspectJudgeError(message)


def _object(value: Any, required: set[str], label: str) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == required, f"invalid {label} fields"
    )
    return cast(dict[str, Any], value)


def _int(value: Any, low: int, high: int, label: str) -> None:
    _require(type(value) is int and low <= value <= high, f"invalid {label}")


def _text(value: Any, maximum: int, label: str) -> None:
    _require(isinstance(value, str) and 0 < len(value) <= maximum, f"invalid {label}")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_credential_fields(value: Any) -> None:
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
                _require(
                    folded not in blocked,
                    "provider request contains a credential field",
                )
                if folded in {"headers", "extra_headers", "http_headers"}:
                    _require(
                        isinstance(child, dict)
                        and {key.casefold() for key in child} <= {"x-irid"}
                        and all(
                            isinstance(item, str) and 0 < len(item) <= 128
                            for item in child.values()
                        ),
                        "provider request contains unsupported headers",
                    )
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)


@dataclass(frozen=True)
class CollectionOptions:
    """Explicit resource reservations; none of these values inherit SDK defaults."""

    grader: str
    inspect_version: str
    profile: str
    epochs: int
    log_model_api: bool
    log_samples: bool
    sdk_max_retries: int
    tools: bool
    concurrency: int
    requests_per_minute: int
    request_timeout_seconds: int
    max_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_cost_microusd: int
    input_tokens_per_call: int
    cost_microusd_per_call: int

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> CollectionOptions:
        _require(isinstance(value, dict), "collection options must be an object")
        _text(value.get("grader"), 256, "explicit grader")
        _object(value, {field.name for field in fields(cls)}, "collection options")
        result = cls(**value)
        result.validate()
        return result

    def validate(self) -> None:
        _text(self.grader, 256, "explicit grader")
        # The named model is an identity, never a URL or a credential carrier.
        _require(
            all(part not in self.grader for part in (":", "@", "?", "#", "\\"))
            and all(ord(char) >= 33 for char in self.grader),
            "grader must be an explicit model identity without URL credentials",
        )
        _require(
            self.inspect_version in REPLAY_INSPECT_VERSIONS,
            "unsupported Inspect version",
        )
        _require(self.profile == EXPORT_PROFILE, "unsupported Inspect profile")
        _require(
            type(self.epochs) is int and self.epochs == 1, "only epoch one is supported"
        )
        _require(
            self.log_model_api is True and self.log_samples is True,
            "complete API and sample logging is required",
        )
        _require(
            type(self.sdk_max_retries) is int and self.sdk_max_retries == 0,
            "SDK retries must be disabled",
        )
        _require(self.tools is False, "tools are unsupported")
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
            _int(getattr(self, name), 1, maximum, name)


def _check_options(plan: JudgeMeasurementPlan, options: CollectionOptions) -> str:
    digest = measurement_plan_digest(plan)
    options.validate()
    _require(
        options.grader == plan["judge"]["requested_model"],
        "grader differs from approved plan",
    )
    _require(
        plan["judge"]["model_identity"]["kind"] == "hosted_api",
        "local weight execution is not qualified by this adapter",
    )
    if options.grader == "openai/gpt-5.6-sol":
        _require(
            options.inspect_version == "0.3.263"
            and Decimal(plan["judge"]["config"]["temperature"]) == Decimal(1),
            "GPT-5.6 Sol requires Inspect 0.3.263 and approved temperature 1",
        )
    return digest


def render_request(
    plan: JudgeMeasurementPlan, *, input_text: str, answer: str
) -> dict[str, Any]:
    """Return the core normalized request with data isolated in JSON fields.

    Templates are literal instructions, never executable formatting expressions.
    JSON field separation does not establish prompt injection immunity.
    """
    measurement_plan_digest(plan)
    return _render_request(plan, input_text=input_text, answer=answer)


def _render_request(
    plan: JudgeMeasurementPlan, *, input_text: str, answer: str
) -> dict[str, Any]:
    """Render after the caller has validated the plan once."""
    _require(
        isinstance(input_text, str)
        and len(input_text.encode("utf-8")) <= JUDGE_REQUEST_MAX_BYTES,
        "input must be bounded text",
    )
    _require(
        isinstance(answer, str)
        and len(answer.encode("utf-8")) <= JUDGE_REQUEST_MAX_BYTES,
        "answer must be bounded text",
    )
    content = render_judge_request(plan, input_text=input_text, answer_text=answer)
    _require(
        len(content) <= JUDGE_REQUEST_MAX_BYTES,
        "rendered request exceeds byte allowance",
    )
    return cast(
        dict[str, Any], parse_json_bytes(content, label="normalized judge request")
    )


def bind_requests(
    plan: JudgeMeasurementPlan, frozen_inputs: dict[str, dict[str, str]]
) -> JudgeMeasurementPlan:
    """Return a plan with exact rendered-request pins for independent approval.

    This preparation helper changes the plan digest. Its return value must be
    approved before collection; it does not mutate an already approved design.
    """
    result = copy.deepcopy(plan)
    _require(isinstance(frozen_inputs, dict), "frozen inputs must be an object")
    _require(
        set(frozen_inputs)
        == {binding["case_id"] for binding in result["answer_bindings"]},
        "frozen input membership differs from plan",
    )
    measurement_plan_digest(result)
    for binding in cast(dict[str, Any], result)["answer_bindings"]:
        for side in ("baseline", "subject"):
            binding[f"{side}_request_sha256"] = "0" * 64
    for binding in cast(dict[str, Any], result)["answer_bindings"]:
        row = _object(
            frozen_inputs[binding["case_id"]],
            {"input", "baseline", "subject"},
            "frozen input",
        )
        for side in ("baseline", "subject"):
            _require(
                isinstance(row[side], str)
                and _sha(row[side].encode()) == binding[f"{side}_answer_sha256"],
                "frozen answer digest mismatch",
            )
            request = _render_request(result, input_text=row["input"], answer=row[side])
            binding[f"{side}_request_sha256"] = _sha(canonical_payload(request))
    measurement_plan_digest(result)
    return result


def prepare_inspect_config(
    plan: JudgeMeasurementPlan, options: CollectionOptions
) -> Any:
    """Construct the pinned SDK's GenerateConfig; never select or call a model."""
    _check_options(plan, options)
    _require(
        options.inspect_version == INSPECT_VERSION,
        "live collection requires current Inspect version",
    )
    try:
        version = importlib.metadata.version("inspect-ai")
    except importlib.metadata.PackageNotFoundError as exc:
        raise InspectJudgeError(
            "install the inspect extra to construct SDK configuration"
        ) from exc
    _require(version == INSPECT_VERSION, "unsupported installed Inspect version")
    module = importlib.import_module("inspect_ai.model")
    config = plan["judge"]["config"]
    return module.GenerateConfig(
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        max_tokens=config["max_output_tokens"],
        seed=config["seed"],
        max_retries=0,
        timeout=options.request_timeout_seconds,
        attempt_timeout=options.request_timeout_seconds,
        max_connections=options.concurrency,
        adaptive_connections=False,
        num_choices=1,
        internal_tools=False,
        parallel_tool_calls=False,
        reasoning_summary="none",
        reasoning_history="none",
        cache=False,
        batch=False,
    )


def prepare_collection(
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    *,
    checkpoint: JudgeMeasurements | None = None,
    baseline_run: dict[str, Any] | None = None,
    subject_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reserve a bounded next batch, reusing terminal checkpoint outcomes.

    This pure planner does not sleep, issue calls, persist checkpoints, or enforce
    provider billing. Reservations are declared upper bounds, not token estimates.
    """
    digest = _check_options(plan, options)
    retained: dict[str, Any] = {}
    spent_calls = 0
    if checkpoint is not None:
        _require(
            isinstance(baseline_run, dict) and isinstance(subject_run, dict),
            "checkpoint resume requires both frozen runs",
        )
        validate_measurements(
            checkpoint,
            plan,
            baseline_run=cast(dict[str, Any], baseline_run),
            subject_run=cast(dict[str, Any], subject_run),
        )
        retained = {trial["trial_id"]: trial for trial in checkpoint["trials"]}
        spent_calls = sum(len(trial["attempts"]) for trial in checkpoint["trials"])
    used_input = spent_calls * options.input_tokens_per_call
    used_output = spent_calls * plan["judge"]["config"]["max_output_tokens"]
    used_cost = spent_calls * options.cost_microusd_per_call
    capacity = min(
        options.max_calls - spent_calls,
        (options.max_input_tokens - used_input) // options.input_tokens_per_call,
        (options.max_output_tokens - used_output)
        // plan["judge"]["config"]["max_output_tokens"],
        (options.max_cost_microusd - used_cost) // options.cost_microusd_per_call,
    )
    pending: list[dict[str, Any]] = []
    terminal = 0
    for binding in sorted(plan["answer_bindings"], key=lambda item: item["case_id"]):
        for side in ("baseline", "subject"):
            for repetition in range(1, plan["schedule"]["repetitions"] + 1):
                trial_id = expected_trial_id(
                    digest, binding["case_id"], side, repetition
                )
                prior = retained.get(trial_id)
                attempts = prior["attempts"] if prior else []
                if attempts and (
                    attempts[-1]["status"] != "transport_error"
                    or len(attempts) >= plan["schedule"]["max_attempts"]
                ):
                    terminal += 1
                    continue
                pending.append(
                    {
                        "trial_id": trial_id,
                        "case_id": binding["case_id"],
                        "side": side,
                        "repetition": repetition,
                        "attempt": len(attempts) + 1,
                    }
                )
    if checkpoint is None:
        empty_trials = []
        bindings = {item["case_id"]: item for item in plan["answer_bindings"]}
        for item in pending:
            empty_binding = cast(dict[str, Any], bindings[item["case_id"]])
            empty_trials.append(
                {
                    "trial_id": item["trial_id"],
                    "case_id": item["case_id"],
                    "side": item["side"],
                    "repetition": item["repetition"],
                    "answer_sha256": empty_binding[f"{item['side']}_answer_sha256"],
                    "plan_sha256": digest,
                    "status": "incomplete",
                    "attempts": [],
                    "selected_attempt": None,
                    "parse": {"status": "unavailable", "rating": None, "value": None},
                }
            )
        checkpoint = cast(
            JudgeMeasurements,
            _assemble_measurements(
                digest, empty_trials, [{"events": []} for _ in empty_trials], options
            ),
        )
    retained_bytes = len(canonical_payload(checkpoint))
    remapping_allowance = plan["schedule"]["expected_trials"] * 128
    storage_capacity = min(
        MAX_SOURCES - len(checkpoint["sources"]),
        (MEASUREMENTS_MAX_BYTES - retained_bytes - remapping_allowance)
        // MAX_ADMISSION_GROWTH_BYTES,
    )
    capacity = min(capacity, storage_capacity)
    admitted = min(len(pending), max(0, capacity), options.concurrency)
    return {
        "plan_sha256": digest,
        "options": asdict(options),
        "spent_calls": spent_calls,
        "terminal_trials": terminal,
        "pending_trials": len(pending),
        "next_batch": pending[:admitted],
        "budget_exhausted": bool(pending) and capacity <= 0,
        "storage_reservation": {
            "retained_bytes": retained_bytes,
            "maximum_bytes": MEASUREMENTS_MAX_BYTES,
            "bytes_per_admitted_call": MAX_ADMISSION_GROWTH_BYTES,
            "source_count": len(checkpoint["sources"]),
            "maximum_sources": MAX_SOURCES,
        },
        "minimum_request_spacing_seconds": 60 / options.requests_per_minute,
    }


def _blob(value: Any) -> dict[str, Any]:
    content = canonical_payload(value)
    _require(len(content) <= 1048576, "request or response exceeds byte allowance")
    return {
        "media_type": "application/json",
        "text": content.decode(),
        "sha256": _sha(content),
    }


def _completion_value(completion: Any) -> Any:
    _require(isinstance(completion, str), "model completion must be text")
    _require(
        len(completion.encode("utf-8")) <= 1048576,
        "model completion exceeds byte allowance",
    )
    try:
        return parse_json_bytes(completion.encode("utf-8"), label="judge completion")
    except StrictJsonError:
        return completion


def _parse_rating(response: Any, plan: JudgeMeasurementPlan) -> dict[str, Any]:
    ratings = {rating["label"]: rating["value"] for rating in plan["scale"]["ratings"]}
    if (
        isinstance(response, dict)
        and set(response) == {"rating"}
        and isinstance(response["rating"], str)
        and response["rating"] in ratings
    ):
        return {
            "status": "ok",
            "rating": response["rating"],
            "value": ratings[response["rating"]],
        }
    return {"status": "invalid", "rating": None, "value": None}


def _assemble_measurements(
    digest: str,
    trials: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    options: CollectionOptions,
) -> dict[str, Any]:
    """Greedily shard whole records, preserving deterministic local positions."""
    sources: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    source_id = "inspect-export"

    def document(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "format": RETAINED_SOURCE_FORMAT,
            "inspect_version": options.inspect_version,
            "collection": asdict(options),
            "records": rows,
        }

    header_bytes = len(canonical_payload(document([])))
    source_bytes = header_bytes

    def publish() -> None:
        _require(
            len(sources) < MAX_SOURCES, "retained sources exceed the source allowance"
        )
        payload = canonical_payload(document(records))
        _require(
            len(payload) <= MAX_SOURCE_BYTES, "retained source exceeds byte allowance"
        )
        sources.append(
            {
                "source_id": source_id,
                "profile": RETAINED_SOURCE_PROFILE,
                "encoding": "utf-8",
                "byte_size": len(payload),
                "media_type": "application/json",
                "content": payload.decode(),
                "sha256": _sha(payload),
            }
        )

    for trial, sample in zip(trials, samples, strict=True):
        for attempt in trial["attempts"]:
            attempt["source"].update(source_id=source_id, record_index=len(records))
        record = {"trial": trial, "events": sample["events"]}
        record_bytes = len(canonical_payload(record))
        if records and source_bytes + 1 + record_bytes > MAX_SOURCE_BYTES:
            publish()
            records = []
            source_bytes = header_bytes
            source_id = f"inspect-export-{len(sources) + 1:04d}"
            for attempt in trial["attempts"]:
                attempt["source"].update(source_id=source_id, record_index=0)
            record_bytes = len(canonical_payload(record))
        _require(
            source_bytes + bool(records) + record_bytes <= MAX_SOURCE_BYTES,
            "one retained trial exceeds the per-source byte allowance",
        )
        source_bytes += bool(records) + record_bytes
        records.append(record)
    if records:
        publish()
    completed = sum(trial["status"] == "complete" for trial in trials)
    measurements = {
        "format": "invarlock/judge-measurements-v1",
        "profile_id": "text-frozen-answer-v1",
        "plan_sha256": digest,
        "source_profile": RETAINED_SOURCE_PROFILE,
        "sources": sources,
        "trials": trials,
        "completeness": {
            "status": "complete" if completed == len(trials) else "incomplete",
            "expected_trials": len(trials),
            "recorded_trials": len(trials),
            "completed_trials": completed,
        },
    }
    _require(
        len(canonical_payload(measurements)) <= MEASUREMENTS_MAX_BYTES,
        "retained measurements exceed the aggregate byte allowance",
    )
    return measurements


def import_export(
    payload: bytes,
    *,
    plan: JudgeMeasurementPlan,
    options: CollectionOptions,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
) -> JudgeMeasurements:
    """Import the explicit expanded-event projection, without Inspect or network.

    The input is this adapter's bounded projection of ModelEvent and ModelCall,
    not an arbitrary Inspect .eval archive. Every scheduled slot must be present;
    unattempted slots use an empty events list and remain incomplete.
    """
    digest = _check_options(plan, options)
    _require(
        isinstance(payload, bytes) and len(payload) <= MAX_EXPORT_BYTES,
        "export exceeds byte allowance",
    )
    try:
        exported = parse_json_bytes(payload, label="Inspect judge export")
    except StrictJsonError as exc:
        raise InspectJudgeError("invalid Inspect judge export JSON") from exc
    root = _object(
        exported,
        {"format", "profile", "inspect_version", "collection", "samples"},
        "export",
    )
    _require(
        root["format"] == EXPORT_FORMAT and root["profile"] == EXPORT_PROFILE,
        "unsupported export profile",
    )
    _require(
        root["inspect_version"] == options.inspect_version,
        "unsupported Inspect version",
    )
    _require(
        canonical_payload(root["collection"]) == canonical_payload(asdict(options)),
        "export collection settings differ from explicit options",
    )
    _require(
        isinstance(baseline_run, dict) and isinstance(subject_run, dict),
        "import requires both frozen runs",
    )
    frozen_inputs: dict[str, dict[str, str]] = {}
    for side, run in (("baseline", baseline_run), ("subject", subject_run)):
        _require(isinstance(run.get("records"), list), "frozen run requires records")
        seen: set[str] = set()
        for record in run["records"]:
            _require(
                isinstance(record, dict) and isinstance(record.get("id"), str),
                "invalid frozen record",
            )
            case_id = record["id"]
            _require(case_id not in seen, "duplicate frozen case")
            seen.add(case_id)
            _require(
                isinstance(record.get("input"), str)
                and isinstance(record.get("output"), str)
                and record.get("error") is None,
                "frozen records require successful text answers",
            )
            row = frozen_inputs.setdefault(case_id, {"input": record["input"]})
            _require(row["input"] == record["input"], "frozen paired inputs differ")
            row[side] = record["output"]
    bindings = {
        item["case_id"]: cast(dict[str, Any], item) for item in plan["answer_bindings"]
    }
    _require(
        set(frozen_inputs) == set(bindings), "frozen input membership differs from plan"
    )
    for case_id, row in frozen_inputs.items():
        _object(row, {"input", "baseline", "subject"}, "frozen input")
        for side in ("baseline", "subject"):
            _require(
                isinstance(row[side], str)
                and _sha(row[side].encode())
                == bindings[case_id][f"{side}_answer_sha256"],
                "frozen answer digest mismatch",
            )
    samples = root["samples"]
    _require(
        isinstance(samples, list)
        and len(samples) == plan["schedule"]["expected_trials"],
        "export must retain every scheduled slot",
    )
    trials = []
    event_ids: set[str] = set()
    for record_index, sample in enumerate(samples):
        sample = _object(sample, {"id", "epoch", "metadata", "events"}, "sample")
        _require(
            type(sample["epoch"]) is int and sample["epoch"] == 1,
            "only epoch one is supported",
        )
        metadata = _object(
            sample["metadata"],
            {
                "case_id",
                "side",
                "repetition",
                "plan_sha256",
                "answer_sha256",
                "scorer_id",
            },
            "sample metadata",
        )
        case_id, side, repetition = (
            metadata["case_id"],
            metadata["side"],
            metadata["repetition"],
        )
        _require(
            isinstance(case_id, str)
            and case_id in bindings
            and side in ("baseline", "subject"),
            "unknown sample binding",
        )
        _int(repetition, 1, plan["schedule"]["repetitions"], "repetition")
        trial_id = expected_trial_id(digest, case_id, side, repetition)
        _require(
            sample["id"] == trial_id
            and metadata["plan_sha256"] == digest
            and metadata["answer_sha256"] == bindings[case_id][f"{side}_answer_sha256"],
            "sample binding mismatch",
        )
        _text(metadata["scorer_id"], 128, "scorer identity")
        expected_request = _render_request(
            plan,
            input_text=frozen_inputs[case_id]["input"],
            answer=frozen_inputs[case_id][side],
        )
        _require(
            _sha(canonical_payload(expected_request))
            == bindings[case_id][f"{side}_request_sha256"],
            "rendered request digest differs from approved plan",
        )
        events = sample["events"]
        _require(
            isinstance(events, list)
            and len(events) <= plan["schedule"]["max_attempts"],
            "invalid attempt count",
        )
        attempts = []
        parsed = {"status": "unavailable", "rating": None, "value": None}
        selected = None
        for index, event in enumerate(events):
            _require(
                len(canonical_payload(event)) <= MAX_RETAINED_EVENT_BYTES,
                "model event exceeds the retained-event byte allowance",
            )
            event = _object(
                event,
                {
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
                },
                "model event",
            )
            _text(event["uuid"], 128, "model event ID")
            _require(event["uuid"] not in event_ids, "duplicate model event ID")
            event_ids.add(event["uuid"])
            _require(
                event["event"] == "model"
                and event["role"] == "grader"
                and event["model"] == options.grader,
                "explicit grader event is required",
            )
            _require(
                event["tools"] == [] and event["tool_choice"] == "none",
                "tools are unsupported",
            )
            _require(
                type(event["retries"]) is int
                and event["retries"] == 0
                and event["cache"] is None,
                "SDK retries and caching are unsupported",
            )
            expected_config = {
                "temperature": float(plan["judge"]["config"]["temperature"]),
                "top_p": float(plan["judge"]["config"]["top_p"]),
                "max_tokens": plan["judge"]["config"]["max_output_tokens"],
                "seed": plan["judge"]["config"]["seed"],
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
            _require(
                canonical_payload(event["config"])
                == canonical_payload(expected_config),
                "model event has hidden or changed generation settings",
            )
            _require(
                event["input"] == expected_request["messages"],
                "model input differs from frozen request",
            )
            call = _object(
                event["call"], {"request", "response", "error"}, "full API call"
            )
            _require(
                isinstance(call["request"], dict)
                and len(canonical_payload(call["request"])) <= 1048576,
                "full API request must be a bounded object",
            )
            _reject_credential_fields(call["request"])
            output = _object(
                event["output"],
                {"model", "request_id", "finish_reason", "usage", "completion"},
                "accessible output",
            )
            error = event["error"]
            response = call["response"]
            if error is None:
                _require(
                    call["error"] in (False, None) and isinstance(response, dict),
                    "completed call requires full response",
                )
                _require(
                    output["model"] in plan["judge"]["approved_resolved_models"],
                    "unapproved resolved model",
                )
                status = "completed"
                accessible_response = _completion_value(output["completion"])
                parsed = _parse_rating(accessible_response, plan)
                selected = index + 1
            else:
                error = _object(error, {"status", "code", "message"}, "attempt error")
                _require(
                    isinstance(error["status"], str)
                    and error["status"]
                    in {"transport_error", "timeout_ambiguous", "cancelled", "refusal"}
                    and call["error"] is True,
                    "unknown attempt error",
                )
                status = error["status"]
                _require(
                    response is None or isinstance(response, dict),
                    "failed API response must be an object or null",
                )
                error = {"code": error["code"], "message": error["message"]}
                parsed = {
                    "status": "refusal" if status == "refusal" else "unavailable",
                    "rating": None,
                    "value": None,
                }
            usage = output["usage"]
            if usage is not None:
                usage = _object(usage, {"input_tokens", "output_tokens"}, "token usage")
                _int(
                    usage["input_tokens"],
                    0,
                    options.input_tokens_per_call,
                    "input token usage",
                )
                _int(
                    usage["output_tokens"],
                    0,
                    plan["judge"]["config"]["max_output_tokens"],
                    "output token usage",
                )
            _require(
                status != "completed" or usage is not None,
                "completed call requires token usage",
            )
            if index + 1 < len(events):
                _require(
                    status == "transport_error"
                    and "transport_error" in plan["schedule"]["retry_on"],
                    "cannot retry a terminal judgment",
                )
            attempts.append(
                {
                    "attempt": index + 1,
                    "role": "judge",
                    "resolved_model": output["model"],
                    "status": status,
                    "request": _blob(expected_request),
                    "response": (
                        _blob(_completion_value(output["completion"]))
                        if status == "completed"
                        else None
                    ),
                    "request_id": output["request_id"],
                    "finish_reason": output["finish_reason"],
                    "error": error,
                    "usage": usage,
                    "cache": "none",
                    "source": {
                        "source_id": "inspect-export",
                        "scorer_id": metadata["scorer_id"],
                        "model_event_id": event["uuid"],
                        "record_index": record_index,
                        "attempt_index": index,
                    },
                }
            )
        trials.append(
            {
                "trial_id": trial_id,
                "case_id": case_id,
                "side": side,
                "repetition": repetition,
                "answer_sha256": metadata["answer_sha256"],
                "plan_sha256": digest,
                "status": "complete" if parsed["status"] == "ok" else "incomplete",
                "attempts": attempts,
                "selected_attempt": selected,
                "parse": parsed,
            }
        )
    spent_calls = sum(len(trial["attempts"]) for trial in trials)
    _require(spent_calls <= options.max_calls, "export exceeds the call allowance")
    _require(
        spent_calls * options.input_tokens_per_call <= options.max_input_tokens,
        "export exceeds the reserved input-token allowance",
    )
    _require(
        spent_calls * plan["judge"]["config"]["max_output_tokens"]
        <= options.max_output_tokens,
        "export exceeds the reserved output-token allowance",
    )
    _require(
        spent_calls * options.cost_microusd_per_call <= options.max_cost_microusd,
        "export exceeds the reserved cost allowance",
    )
    measurements = _assemble_measurements(digest, trials, samples, options)
    result = cast(JudgeMeasurements, measurements)
    validate_measurements(
        result, plan, baseline_run=baseline_run, subject_run=subject_run
    )
    return result
