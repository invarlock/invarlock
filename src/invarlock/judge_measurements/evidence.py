"""Publish and replay retained judge evidence without conferring acceptance."""

from __future__ import annotations

import base64
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jsonschema import Draft202012Validator

from invarlock.captured_contracts import secure_directory
from invarlock.evaluation_record_contracts.contracts import MAX_INPUT_BYTES
from invarlock.evaluation_records.cases import case_set_digest
from invarlock.evaluation_records.io import run_digest
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes
from invarlock.filesystem.atomic_directory import publish_directory_no_replace
from invarlock.filesystem.atomic_file import write_file_no_replace
from invarlock.judge_evidence_types import JudgeEvidenceBindings, JudgeEvidenceEnvelope
from invarlock.judge_measurement_types import (
    JudgeAnalysisPolicyDocument,
    JudgeMeasurementPlan,
    JudgeMeasurements,
)
from invarlock.judge_measurements.analysis import (
    ANALYSIS_POLICY_MAX_BYTES,
    JudgeAnalysisResult,
    analyze_measurements,
    decode_analysis_policy,
)
from invarlock.judge_measurements.contracts import (
    MEASUREMENTS_MAX_BYTES,
    PLAN_MAX_BYTES,
    canonical_payload,
    measurement_plan_digest,
)
from invarlock.public_contracts import load_judge_measurement_evidence_schema

DECISION_SCOPE: Final = "bounded-judge-fixed-benchmark-v1"
EVIDENCE_FORMAT: Final = "invarlock/judge-measurement-evidence-v1"
ARTIFACT_FILENAMES = (
    "plan.json",
    "measurements.json",
    "baseline_run.json",
    "subject_run.json",
    "case_set.json",
    "analysis_policy.json",
    "analysis_result.json",
)
_MAX_ENVELOPE_BYTES = 64 * 1024
ANALYSIS_RESULT_MAX_BYTES = 1024 * 1024
ARTIFACT_BYTE_LIMITS = {
    "plan.json": PLAN_MAX_BYTES,
    "measurements.json": MEASUREMENTS_MAX_BYTES,
    "baseline_run.json": MAX_INPUT_BYTES,
    "subject_run.json": MAX_INPUT_BYTES,
    "case_set.json": MAX_INPUT_BYTES,
    "analysis_policy.json": ANALYSIS_POLICY_MAX_BYTES,
    "analysis_result.json": ANALYSIS_RESULT_MAX_BYTES,
    "envelope.json": _MAX_ENVELOPE_BYTES,
}


class JudgeEvidenceError(ValueError):
    """Retained measurement evidence could not be safely replayed or published."""


@dataclass(frozen=True)
class JudgeEvidencePublication:
    path: Path
    envelope: JudgeEvidenceEnvelope
    analysis_result: JudgeAnalysisResult


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_payload(value)).hexdigest()


def signed_envelope_bytes(envelope: JudgeEvidenceEnvelope) -> bytes:
    """Domain-separate the exact envelope statement from other signed formats."""
    statement = {key: value for key, value in envelope.items() if key != "signature"}
    return EVIDENCE_FORMAT.encode("ascii") + b"\0" + canonical_payload(statement)


def read_object(path: Path, *, maximum: int | None = None) -> dict[str, Any]:
    if maximum is None:
        maximum = ARTIFACT_BYTE_LIMITS.get(path.name, MAX_INPUT_BYTES)
    value = parse_json_bytes(
        read_regular_file_bytes(path, label=path.name, max_bytes=maximum),
        label=path.name,
    )
    if not isinstance(value, dict):
        raise JudgeEvidenceError(f"{path.name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _validate_envelope(envelope: JudgeEvidenceEnvelope) -> None:
    error = next(
        Draft202012Validator(load_judge_measurement_evidence_schema()).iter_errors(
            envelope
        ),
        None,
    )
    if error is not None:
        raise JudgeEvidenceError(
            f"judge evidence envelope is invalid: {error.message[:240]}"
        )


def load_judge_evidence_envelope(path: Path) -> JudgeEvidenceEnvelope:
    """Read only the small, closed envelope before authorizing expensive replay."""
    root = Path(path).absolute()
    with secure_directory(root):
        envelope = cast(JudgeEvidenceEnvelope, read_object(root / "envelope.json"))
    _validate_envelope(envelope)
    return envelope


def _private_key(value: Path | Ed25519PrivateKey) -> Ed25519PrivateKey:
    if isinstance(value, Ed25519PrivateKey):
        return value
    key = serialization.load_pem_private_key(
        read_regular_file_bytes(
            Path(value), label="judge evidence signing key", max_bytes=65536
        ),
        password=None,
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise JudgeEvidenceError("judge evidence signing key must be Ed25519")
    return key


def _analyze(artifacts: dict[str, dict[str, Any]]) -> JudgeAnalysisResult:
    plan = cast(JudgeMeasurementPlan, artifacts["plan.json"])
    policy = decode_analysis_policy(
        cast(JudgeAnalysisPolicyDocument, artifacts["analysis_policy.json"]), plan=plan
    )
    return analyze_measurements(
        plan,
        cast(JudgeMeasurements, artifacts["measurements.json"]),
        policy,
        baseline_run=artifacts["baseline_run.json"],
        subject_run=artifacts["subject_run.json"],
    )


def _bindings(artifacts: dict[str, dict[str, Any]]) -> JudgeEvidenceBindings:
    return {
        "baseline_run_sha256": run_digest(artifacts["baseline_run.json"]),
        "subject_run_sha256": run_digest(artifacts["subject_run.json"]),
        "case_set_sha256": case_set_digest(artifacts["case_set.json"]),
        "plan_sha256": measurement_plan_digest(
            cast(JudgeMeasurementPlan, artifacts["plan.json"])
        ),
        "measurements_sha256": object_sha256(artifacts["measurements.json"]),
        "analysis_policy_sha256": object_sha256(artifacts["analysis_policy.json"]),
        "analysis_result_sha256": object_sha256(artifacts["analysis_result.json"]),
    }


def publish_judge_evidence(
    output: Path,
    *,
    plan: JudgeMeasurementPlan,
    measurements: JudgeMeasurements,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
    analysis_policy: JudgeAnalysisPolicyDocument,
    signing_key: Path | Ed25519PrivateKey | None = None,
    signer_identity: str | None = None,
) -> JudgeEvidencePublication:
    """Replay before atomically publishing a new, immutable evidence directory.

    An unsigned envelope remains useful for inspection but cannot satisfy the
    separate recipient verifier. This operation does not create recipient trust.
    """
    if (signing_key is None) != (signer_identity is None):
        raise JudgeEvidenceError(
            "signing_key and signer_identity must be supplied together"
        )
    artifacts = cast(
        dict[str, dict[str, Any]],
        parse_json_bytes(
            canonical_payload(
                {
                    "plan.json": plan,
                    "measurements.json": measurements,
                    "baseline_run.json": baseline_run,
                    "subject_run.json": subject_run,
                    "analysis_policy.json": analysis_policy,
                    "case_set.json": {
                        "format": "invarlock/evaluation-case-set-v1",
                        "cases": [
                            {
                                key: record[key]
                                for key in ("id", "input", "expected", "metadata")
                            }
                            for record in baseline_run["records"]
                        ],
                    },
                }
            ),
            label="judge evidence artifacts",
        ),
    )
    result = _analyze(artifacts)
    artifacts["analysis_result.json"] = result.to_dict()
    envelope: JudgeEvidenceEnvelope = {
        "format": EVIDENCE_FORMAT,
        "decision_scope": DECISION_SCOPE,
        "intended_subject": artifacts["subject_run.json"]["artifact_digest"],
        "bindings": _bindings(artifacts),
        "signature_algorithm": "ed25519",
        "signer": None,
        "signature": None,
    }
    if signing_key is not None:
        key = _private_key(signing_key)
        public_key = key.public_key()
        envelope["signer"] = {
            "identity": cast(str, signer_identity),
            "public_key_sha256": public_key_fingerprint(public_key),
            "public_key": base64.b64encode(public_key.public_bytes_raw()).decode(
                "ascii"
            ),
        }
        envelope["signature"] = base64.b64encode(
            key.sign(signed_envelope_bytes(envelope))
        ).decode("ascii")
    _validate_envelope(envelope)
    destination = Path(output).absolute()
    with secure_directory(destination.parent, create=True):
        with tempfile.TemporaryDirectory(
            prefix=".judge-evidence-", dir=destination.parent
        ) as stage_name:
            stage = Path(stage_name)
            for name, artifact in artifacts.items():
                payload = canonical_payload(artifact) + b"\n"
                if len(payload) > ARTIFACT_BYTE_LIMITS[name]:
                    raise JudgeEvidenceError(
                        f"{name} exceeds the {ARTIFACT_BYTE_LIMITS[name]}-byte size limit"
                    )
                write_file_no_replace(
                    stage / name,
                    payload,
                    create_parents=False,
                )
            write_file_no_replace(
                stage / "envelope.json",
                canonical_payload(envelope) + b"\n",
                create_parents=False,
            )
            publish_directory_no_replace(stage, destination)
    return JudgeEvidencePublication(destination, envelope, result)


def replay_judge_evidence(
    path: Path, *, expected_envelope_sha256: str | None = None
) -> JudgeEvidencePublication:
    """Replay fixed artifacts and signed bindings; signer authorization is separate."""
    root = Path(path).absolute()
    with secure_directory(root):
        envelope = cast(JudgeEvidenceEnvelope, read_object(root / "envelope.json"))
        _validate_envelope(envelope)
        if (
            expected_envelope_sha256 is not None
            and object_sha256(envelope) != expected_envelope_sha256
        ):
            raise JudgeEvidenceError("judge evidence envelope changed before replay")
        artifacts = {name: read_object(root / name) for name in ARTIFACT_FILENAMES}
    actual = _bindings(artifacts)
    if envelope["bindings"] != actual:
        raise JudgeEvidenceError(
            "judge evidence artifact bindings do not match the envelope"
        )
    if actual["case_set_sha256"] != artifacts["plan.json"]["case_set_sha256"]:
        raise JudgeEvidenceError(
            "judge evidence case set does not match the approved plan"
        )
    if envelope["intended_subject"] != artifacts["subject_run.json"]["artifact_digest"]:
        raise JudgeEvidenceError(
            "judge evidence intended subject does not match the frozen run"
        )
    result = _analyze(artifacts)
    if canonical_payload(result.to_dict()) != canonical_payload(
        artifacts["analysis_result.json"]
    ):
        raise JudgeEvidenceError(
            "judge evidence analysis result differs from independent replay"
        )
    return JudgeEvidencePublication(root, envelope, result)
