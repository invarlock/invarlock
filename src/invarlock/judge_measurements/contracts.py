"""Semantic validation for retained bounded judge measurements.

JSON Schema closes each wire shape.  This module validates relationships that
schemas cannot express: content digests, fixed schedules, frozen-answer
bindings, retry selection, source replay, and completeness accounting.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

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
from invarlock.public_contracts import (
    load_judge_measurement_plan_schema,
    load_judge_measurements_schema,
)

PLAN_MAX_BYTES = 64 * 1024 * 1024
MEASUREMENTS_MAX_BYTES = 384 * 1024 * 1024
PLAN_FORMAT = "invarlock/judge-measurement-plan-v1"
MEASUREMENTS_FORMAT = "invarlock/judge-measurements-v1"
SOURCE_FORMAT = "invarlock/retained-judge-json-v1"
TRIAL_ID_SCHEME = "plan-case-side-repetition-sha256-v1"


class JudgeMeasurementContractError(ValueError):
    """A judge plan or retained-measurement relationship is invalid."""


def _fail(message: str) -> None:
    raise JudgeMeasurementContractError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(text: str) -> str:
    return _sha256(text.encode("utf-8"))


def canonical_payload(value: object) -> bytes:
    """Return the compact canonical bytes used by judge contract digests."""

    return canonical_json_bytes(value, newline=False)


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
    try:
        size = len(canonical_payload(value))
    except (TypeError, ValueError) as exc:
        raise JudgeMeasurementContractError(f"{label} is not canonical JSON") from exc
    if size > maximum:
        _fail(f"{label} exceeds the {maximum}-byte limit")


def validate_measurement_plan(value: JudgeMeasurementPlan) -> None:
    """Validate the closed plan and all cross-field scheduling invariants."""

    raw = cast(dict[str, Any], value)
    _validate_schema(raw, "plan")
    _bounded_canonical(raw, PLAN_MAX_BYTES, "judge measurement plan")

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
    rating_values: set[str] = set()
    for rating in raw["scale"]["ratings"]:
        label = rating["label"]
        encoded_value = canonical_payload(rating["value"]).decode("ascii")
        if label in ratings:
            _fail("rating labels must be unique")
        if encoded_value in rating_values:
            _fail("rating numeric values must be unique")
        ratings[label] = rating["value"]
        rating_values.add(encoded_value)
    for demonstration in raw["prompt"]["demonstrations"]:
        if demonstration["rating"] not in ratings:
            _fail("every demonstration rating must exist in the declared scale")

    judge = raw["judge"]
    identity = judge["model_identity"]
    if identity["kind"] == "hosted_api" and identity["weights_sha256"] is not None:
        _fail("hosted API judge identity must not claim a weights digest")
    if identity["kind"] == "local_weights" and identity["weights_sha256"] is None:
        _fail("local judge identity requires a weights digest")

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
    if schedule.get("trial_id_scheme") != TRIAL_ID_SCHEME:
        _fail("unsupported judge trial ID scheme")
    expected = len(case_units) * 2 * schedule["repetitions"]
    if schedule["expected_trials"] != expected:
        _fail("expected_trials must equal cases × two sides × repetitions")
    if schedule["max_attempts"] > 1 and schedule["retry_on"] != ["transport_error"]:
        _fail("multiple attempts require transport_error as the sole retry condition")


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
    if _text_sha256(blob["text"]) != blob["sha256"]:
        _fail(f"{label} digest does not match its UTF-8 text")


def _check_attempts(
    trial: dict[str, Any],
    *,
    plan: dict[str, Any],
    source_ids: set[str],
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


def _source_trials(source: dict[str, Any]) -> list[dict[str, Any]]:
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
    if not isinstance(decoded, dict) or set(decoded) != {"format", "trials"}:
        _fail(f"source {source['source_id']!r} has an unsupported retained shape")
    if decoded["format"] != SOURCE_FORMAT or not isinstance(decoded["trials"], list):
        _fail(f"source {source['source_id']!r} has an unsupported retained profile")
    return cast(list[dict[str, Any]], decoded["trials"])


def validate_measurements(value: JudgeMeasurements, plan: JudgeMeasurementPlan) -> None:
    """Replay retained source normalization and validate every planned slot."""

    validate_measurement_plan(plan)
    raw = cast(dict[str, Any], value)
    plan_raw = cast(dict[str, Any], plan)
    _validate_schema(raw, "measurements")
    _bounded_canonical(raw, MEASUREMENTS_MAX_BYTES, "judge measurements")

    plan_sha256 = _sha256(canonical_payload(plan_raw))
    if raw["plan_sha256"] != plan_sha256:
        _fail("measurements do not bind the supplied plan")

    sources: dict[str, dict[str, Any]] = {}
    replayed: dict[str, dict[str, Any]] = {}
    for source in raw["sources"]:
        source_id = source["source_id"]
        if source_id in sources:
            _fail("measurement source IDs must be unique")
        sources[source_id] = source
        for record_index, trial in enumerate(_source_trials(source)):
            trial_id = trial.get("trial_id")
            if not isinstance(trial_id, str) or trial_id in replayed:
                _fail("retained sources must contain unique trial IDs")
            checked_trial_id = cast(str, trial_id)
            for attempt_index, attempt in enumerate(trial.get("attempts", [])):
                mapping = attempt.get("source")
                if not isinstance(mapping, dict) or (
                    mapping.get("source_id") != source_id
                    or mapping.get("record_index") != record_index
                    or mapping.get("attempt_index") != attempt_index
                ):
                    _fail("retained source position mapping is inconsistent")
            replayed[checked_trial_id] = trial

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
    source_positions: set[tuple[str, int]] = set()

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
        _check_attempts(trial_raw, plan=plan_raw, source_ids=set(sources))
        for attempt in trial_raw["attempts"]:
            position = (
                attempt["source"]["source_id"],
                attempt["source"]["record_index"],
            )
            if position in source_positions:
                _fail("source record positions must be unique across trials")
            source_positions.add(position)
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


def load_measurements(path: Path, *, plan: JudgeMeasurementPlan) -> JudgeMeasurements:
    """Read and validate retained measurements from one bounded snapshot."""

    decoded = _load_object(
        Path(path), maximum=MEASUREMENTS_MAX_BYTES, label="judge measurements"
    )
    measurements = cast(JudgeMeasurements, decoded)
    validate_measurements(measurements, plan)
    return measurements


__all__ = [
    "JudgeMeasurementContractError",
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
    "validate_measurement_plan",
    "validate_measurements",
]
