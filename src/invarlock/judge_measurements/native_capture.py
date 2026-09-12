"""Offline replay of exact native runtime evidence into frozen judge answers."""

from __future__ import annotations

import base64
import binascii
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal, cast

from invarlock.core.runtime_provider import build_runtime_behavioral_schedule
from invarlock.evaluation_records.io import run_digest
from invarlock.evidence_pack_contract import (
    MAX_EVIDENCE_BYTES,
    MAX_OBSERVATION_BYTES,
    MAX_OBSERVATIONS,
    EvidenceObservation,
    RuntimeSideEvidence,
    canonical_json_bytes,
    dataset_preparation_binding_errors,
    derive_paired_records,
    evaluation_request_errors,
    parse_json_object,
    sha256_digest,
)
from invarlock.evidence_pack_publication import _preflight_runtime_side, _side_payloads
from invarlock.runtime_provider_evidence import (
    decode_artifact_identity,
    decode_runtime_provider_receipt,
    decode_scoring_observation,
    runtime_request_binding_errors,
)

NATIVE_CAPTURE_FORMAT = "invarlock/native-judge-capture-v1"
NATIVE_RUN_SOURCE = "invarlock-native-judge"
NATIVE_CAPTURE_MAX_BYTES = 128 * 1024 * 1024
_POLICY_MAX_BYTES = 4 * 1024 * 1024
_SIDE_FIELDS = frozenset(field.name for field in fields(RuntimeSideEvidence))
_CAPTURE_FIELDS = frozenset(
    {
        "format",
        "normalized_request",
        "schedule",
        "policy_base64",
        "recipe",
        "baseline",
        "subject",
        "observations",
    }
)


def _decode(value: object, *, maximum: int, label: str) -> bytes:
    if not isinstance(value, str) or len(value) > 4 * ((maximum + 2) // 3):
        raise ValueError(f"native capture {label} exceeds its byte limit or is invalid")
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"native capture {label} is not valid base64") from exc
    if len(payload) > maximum or base64.b64encode(payload).decode("ascii") != value:
        raise ValueError(f"native capture {label} is not bounded canonical base64")
    return payload


def create_native_capture(
    normalized_request: dict[str, Any],
    schedule: dict[str, Any],
    policy_bytes: bytes,
    recipe: dict[str, Any],
    baseline: RuntimeSideEvidence,
    subject: RuntimeSideEvidence,
    observations: tuple[EvidenceObservation, ...] = (),
) -> dict[str, Any]:
    """Freeze all twelve original side files and replay before returning."""
    if len(policy_bytes) > _POLICY_MAX_BYTES:
        raise ValueError("native capture policy exceeds its byte limit")
    raw_sides = [
        getattr(side, name) for side in (baseline, subject) for name in _SIDE_FIELDS
    ]
    if any(
        not isinstance(raw, bytes) or len(raw) > MAX_EVIDENCE_BYTES for raw in raw_sides
    ):
        raise ValueError(
            "native capture side file exceeds its byte limit or is invalid"
        )
    if sum(len(raw) for raw in raw_sides) * 4 // 3 > NATIVE_CAPTURE_MAX_BYTES:
        raise ValueError("native capture exceeds its aggregate byte limit")
    if len(observations) > MAX_OBSERVATIONS:
        raise ValueError("native capture exceeds its observation count limit")
    capture = {
        "format": NATIVE_CAPTURE_FORMAT,
        "normalized_request": normalized_request,
        "schedule": schedule,
        "policy_base64": base64.b64encode(policy_bytes).decode("ascii"),
        "recipe": recipe,
        "observations": [
            {
                "id": observation.observation_id,
                "scope": observation.scope,
                "kind": observation.kind,
                "payload_base64": base64.b64encode(observation.payload).decode("ascii"),
            }
            for observation in sorted(
                observations, key=lambda item: item.observation_id
            )
        ],
        **{
            role: {
                name: base64.b64encode(getattr(side, name)).decode("ascii")
                for name in sorted(_SIDE_FIELDS)
            }
            for role, side in (("baseline", baseline), ("subject", subject))
        },
    }
    # Return an independent snapshot so callers cannot mutate captured inputs.
    frozen = parse_json_object(canonical_json_bytes(capture), label="native capture")
    validate_native_capture(frozen)
    return frozen


def validate_native_capture(
    capture: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate native provenance without loading any runtime provider."""
    if not isinstance(capture, dict) or set(capture) != _CAPTURE_FIELDS:
        raise ValueError("native capture must contain exactly the defined fields")
    if capture["format"] != NATIVE_CAPTURE_FORMAT:
        raise ValueError("native capture format is unsupported")
    capture_bytes = canonical_json_bytes(capture, newline=False)
    if len(capture_bytes) > NATIVE_CAPTURE_MAX_BYTES:
        raise ValueError("native capture exceeds its aggregate byte limit")
    request = capture["normalized_request"]
    if not isinstance(request, dict):
        raise ValueError("native capture normalized request must be an object")
    request_errors = evaluation_request_errors(request)
    if request_errors:
        raise ValueError(request_errors[0])
    schedule = build_runtime_behavioral_schedule(capture["schedule"])
    errors = dataset_preparation_binding_errors(request, schedule)
    if errors:
        raise ValueError(errors[0])
    if any(
        len(record.input_parts) != 1 or record.input_parts[0].kind != "text"
        for record in schedule.records
    ):
        raise ValueError("native judge capture requires exactly one text input part")
    comparison = request["comparison"]
    if comparison.get("metric") != "judge":
        raise ValueError("native judge capture requires a judge metric request")
    if comparison["policy"] != "inputs/policy.json":
        raise ValueError("native capture request must bind the canonical policy")
    policy = _decode(
        capture["policy_base64"], maximum=_POLICY_MAX_BYTES, label="policy"
    )
    recipe = parse_json_object(policy, label="native capture original policy")
    if recipe != capture["recipe"]:
        raise ValueError("native capture recipe differs from its original policy")
    from invarlock.judge_measurements.native_recipe import prepare_native_judge

    prepare_native_judge(policy, capture["schedule"])
    policy_digest = sha256_digest(policy)
    retained = capture["observations"]
    if not isinstance(retained, list) or len(retained) > MAX_OBSERVATIONS:
        raise ValueError("native capture observations must be a bounded array")
    observation_bindings = []
    for value in retained:
        if not isinstance(value, dict) or set(value) != {
            "id",
            "scope",
            "kind",
            "payload_base64",
        }:
            raise ValueError("native capture observation fields are invalid")
        observation = EvidenceObservation(
            observation_id=value["id"],
            scope=cast(Literal["baseline", "subject", "comparison"], value["scope"]),
            kind=value["kind"],
            payload=_decode(
                value["payload_base64"],
                maximum=MAX_OBSERVATION_BYTES,
                label="observation",
            ),
        )
        observation_bindings.append(
            {
                "id": observation.observation_id,
                "scope": observation.scope,
                "kind": observation.kind,
                "payload_digest": sha256_digest(observation.payload),
            }
        )
    ids = [value["id"] for value in observation_bindings]
    if ids != sorted(set(ids)) or observation_bindings != request.get(
        "observations", []
    ):
        raise ValueError(
            "native capture observations differ from the normalized request"
        )
    sides: dict[str, RuntimeSideEvidence] = {}
    runtimes: dict[str, str] = {}
    artifacts: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="invarlock-native-judge-") as temporary:
        stage = Path(temporary)
        for role in ("baseline", "subject"):
            encoded = capture[role]
            if not isinstance(encoded, dict) or set(encoded) != _SIDE_FIELDS:
                raise ValueError(
                    f"native capture {role} must retain all six original files"
                )
            side = RuntimeSideEvidence(
                **{
                    name: _decode(
                        encoded[name],
                        maximum=MAX_EVIDENCE_BYTES,
                        label=f"{role} {name}",
                    )
                    for name in _SIDE_FIELDS
                }
            )
            sides[role] = side
            identity = decode_artifact_identity(side.artifact_identity)
            receipt = decode_runtime_provider_receipt(side.provider_receipt)
            runtime = receipt.outer_image_digest
            if runtime is None:
                raise ValueError(f"native capture {role} lacks strict runtime identity")
            runtimes[role] = runtime
            artifacts[role] = sha256_digest(side.artifact_identity)
            requested = comparison[role]
            model_id = next(
                (
                    getattr(identity, name)
                    for name in ("model_id", "artifact_name", "bundle_name")
                    if hasattr(identity, name)
                ),
                None,
            )
            if requested["artifact"]["model_id"] != model_id:
                raise ValueError(
                    f"native capture {role} model identity differs from request"
                )
            errors = list(
                runtime_request_binding_errors(
                    provider_name=requested["runtime"]["provider"],
                    settings=requested["runtime"]["settings"],
                    artifact_identity=identity,
                    receipt=receipt,
                )
            )
            if errors:
                raise ValueError(f"native capture {role}: {errors[0]}")
            for relative, payload in _side_payloads(role, side).items():
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            _preflight_runtime_side(
                stage,
                side=role,
                runtime_digest=runtime,
                provider_name=receipt.plugin.name,
                schedule_sha256=schedule.schedule_sha256,
                policy_digest=policy_digest,
            )
        derive_paired_records(
            schedule=schedule,
            metric="exact_match",
            baseline=sides["baseline"],
            subject=sides["subject"],
            baseline_identity_digest=artifacts["baseline"],
            subject_identity_digest=artifacts["subject"],
            baseline_runtime_digest=runtimes["baseline"],
            subject_runtime_digest=runtimes["subject"],
        )
    digest = sha256_digest(capture_bytes)
    runs = []
    for role in ("baseline", "subject"):
        scoring = decode_scoring_observation(sides[role].scoring_observation)
        records = [
            {
                "id": scheduled.record_id,
                "input": scheduled.input_text,
                "expected": scheduled.expected_output,
                "output": observed.output_text,
                "scores": {},
                "metadata": {},
                "error": None,
                "context": {
                    "model_id": comparison[role]["artifact"]["model_id"],
                    "provider": scoring.provider_name,
                    "runtime_digest": runtimes[role],
                    "schedule_sha256": schedule.schedule_sha256,
                    "input_sha256": scheduled.input_sha256,
                    "output_sha256": observed.output_sha256,
                },
            }
            for scheduled, observed in zip(
                schedule.records, scoring.records, strict=True
            )
        ]
        run = {
            "format": "invarlock/evaluation-run-v1",
            "source": {"name": NATIVE_RUN_SOURCE, "version": "1"},
            "run_id": f"{role}-{digest.removeprefix('sha256:')}",
            "artifact_digest": artifacts[role],
            "source_digest": digest,
            "score_provenance": {},
            "records": records,
        }
        run_digest(run)
        runs.append(run)
    return runs[0], runs[1]
