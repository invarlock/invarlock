"""Atomic publication for the canonical evidence-pack-v1 envelope."""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock import evidence_pack_integrity as integrity
from invarlock.core.runtime_provider import (
    GGUFArtifactIdentity,
    HFSnapshotArtifactIdentity,
    RuntimeBehavioralSchedule,
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
)
from invarlock.core.scorer_extension import ScorerExtensionRegistry
from invarlock.evidence_pack_contract import (
    EVIDENCE_PACK_FORMAT,
    EVIDENCE_PATHS,
    INPUT_ROLES,
    MAX_EVIDENCE_BYTES,
    MAX_OBSERVATIONS,
    EvidenceObservation,
    EvidencePackError,
    InputIdentity,
    RuntimeSideEvidence,
    build_comparison_report,
    canonical_json_bytes,
    dataset_preparation_binding_errors,
    derive_paired_records,
    evaluation_request_errors,
    evidence_observation_bytes,
    identity_payload,
    normalize_digest,
    parse_json_object,
    request_metric,
    request_scorer_binding,
    runtime_side_config_errors,
    schedule_bytes,
    sha256_digest,
)
from invarlock.evidence_pack_json import StrictJsonError, read_regular_file_bytes
from invarlock.runtime_provider_evidence import (
    RuntimeProviderEvidenceError,
    decode_artifact_identity,
)
from invarlock.runtime_verify import verify_runtime_manifest_snapshot


def _write_new(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(mode)
    except OSError as exc:
        raise EvidencePackError(f"could not write {path.name}: {exc}") from exc


def _load_private_key(path: Path) -> ed25519.Ed25519PrivateKey:
    try:
        payload = read_regular_file_bytes(
            path,
            label="evidence-pack signing key",
            max_bytes=64 * 1024,
        )
        key = serialization.load_pem_private_key(payload, password=None)
    except (StrictJsonError, TypeError, ValueError) as exc:
        raise EvidencePackError(f"could not load signing key: {exc}") from exc
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise EvidencePackError("evidence-pack signing key must be Ed25519")
    return key


def _signature_bytes(
    manifest_bytes: bytes, *, private_key: ed25519.Ed25519PrivateKey
) -> tuple[bytes, str]:
    public_key = private_key.public_key()
    fingerprint = integrity.public_key_fingerprint(public_key)
    payload = {
        "format": integrity.EVIDENCE_PACK_SIGNATURE_FORMAT,
        "algorithm": "ed25519",
        "signing_key_fingerprint": fingerprint,
        "public_key": {
            "encoding": "pem",
            "value": public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("ascii"),
        },
        "signature": {
            "encoding": "base64",
            "value": base64.b64encode(private_key.sign(manifest_bytes)).decode("ascii"),
        },
    }
    return canonical_json_bytes(payload), fingerprint


def _checksum_bytes(files: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {relative}\n"
        for relative, payload in sorted(files.items())
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_no_clobber(staging: Path, destination: Path) -> None:
    lock_path = destination.parent / f".{destination.name}.publish.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise EvidencePackError(
            f"evidence destination is already being published: {destination}"
        ) from exc
    try:
        os.close(descriptor)
        if os.path.lexists(destination):
            raise EvidencePackError(
                f"evidence destination already exists: {destination}"
            )
        try:
            os.rename(staging, destination)
        except OSError as exc:
            raise EvidencePackError(
                f"could not atomically publish evidence pack: {exc}"
            ) from exc
        _fsync_directory(destination.parent)
    finally:
        lock_path.unlink(missing_ok=True)


def _side_payloads(side: str, evidence: RuntimeSideEvidence) -> dict[str, bytes]:
    return {
        EVIDENCE_PATHS[f"{side}_run_report"]: evidence.run_report,
        EVIDENCE_PATHS[f"{side}_runtime_manifest"]: evidence.runtime_manifest,
        EVIDENCE_PATHS[f"{side}_runtime_config"]: evidence.runtime_config,
        EVIDENCE_PATHS[f"{side}_provider_identity"]: evidence.artifact_identity,
        EVIDENCE_PATHS[f"{side}_provider_receipt"]: evidence.provider_receipt,
        EVIDENCE_PATHS[f"{side}_scoring_observation"]: (evidence.scoring_observation),
    }


def _preflight_runtime_side(
    staging: Path,
    *,
    side: str,
    runtime_digest: str,
    provider_name: str,
    schedule_sha256: str,
    policy_digest: str,
) -> None:
    manifest_path = staging / EVIDENCE_PATHS[f"{side}_runtime_manifest"]
    report_path = staging / EVIDENCE_PATHS[f"{side}_run_report"]
    manifest = parse_json_object(
        read_regular_file_bytes(
            manifest_path,
            label=f"{side} runtime manifest",
            max_bytes=MAX_EVIDENCE_BYTES,
        ),
        label=f"{side} runtime manifest",
    )
    result = verify_runtime_manifest_snapshot(
        read_regular_file_bytes(
            report_path, label=f"{side} run report", max_bytes=MAX_EVIDENCE_BYTES
        ),
        manifest,
        report=report_path,
        manifest=manifest_path,
        expected_image_digest=runtime_digest,
        require_strict_runtime=True,
    )
    if result.errors:
        raise EvidencePackError(
            f"{side} runtime snapshot is invalid: " + "; ".join(result.errors)
        )
    try:
        identity = decode_artifact_identity(
            read_regular_file_bytes(
                staging / EVIDENCE_PATHS[f"{side}_provider_identity"],
                label=f"{side} artifact identity",
                max_bytes=MAX_EVIDENCE_BYTES,
            )
        )
    except (RuntimeProviderEvidenceError, StrictJsonError) as exc:
        raise EvidencePackError(str(exc)) from exc
    config_errors = runtime_side_config_errors(
        read_regular_file_bytes(
            staging / EVIDENCE_PATHS[f"{side}_runtime_config"],
            label=f"{side} runtime config",
            max_bytes=MAX_EVIDENCE_BYTES,
        ),
        role=side,
        provider_name=provider_name,
        artifact_identity_sha256=artifact_identity_sha256(identity),
        schedule_sha256=schedule_sha256,
        policy_digest=policy_digest,
    )
    if config_errors:
        raise EvidencePackError(config_errors[0])


def _validate_input_bindings(
    request: Mapping[str, object],
    *,
    schedule: RuntimeBehavioralSchedule,
    identities: Mapping[str, InputIdentity],
    baseline_evidence: RuntimeSideEvidence,
    subject_evidence: RuntimeSideEvidence,
) -> None:
    comparison = request.get("comparison")
    if not isinstance(comparison, Mapping):
        raise EvidencePackError("normalized request comparison is invalid")
    dataset_errors = dataset_preparation_binding_errors(request, schedule)
    if dataset_errors:
        raise EvidencePackError(dataset_errors[0])
    if comparison.get("policy") != "inputs/policy.json":
        raise EvidencePackError(
            "normalized request policy must name the canonical policy identity"
        )
    expected_locators: dict[str, object] = {
        "dataset": EVIDENCE_PATHS["schedule"],
        "policy": "inputs/policy.json",
        "baseline_runtime": f"runtime:{identities['baseline_runtime'].digest}",
        "subject_runtime": f"runtime:{identities['subject_runtime'].digest}",
    }
    side_evidence = {
        "baseline": baseline_evidence,
        "subject": subject_evidence,
    }
    for side in ("baseline", "subject"):
        side_request = comparison.get(side)
        artifact_request = (
            side_request.get("artifact") if isinstance(side_request, Mapping) else None
        )
        if not isinstance(artifact_request, Mapping):
            raise EvidencePackError(f"normalized request {side} artifact is invalid")
        expected_locators[side] = artifact_request.get("locator")
        try:
            artifact = decode_artifact_identity(side_evidence[side].artifact_identity)
        except RuntimeProviderEvidenceError as exc:
            raise EvidencePackError(
                f"{side} provider artifact identity is invalid: {exc}"
            ) from exc
        if isinstance(artifact, HFSnapshotArtifactIdentity):
            authenticated_model_id = artifact.model_id
        elif isinstance(artifact, GGUFArtifactIdentity):
            authenticated_model_id = artifact.artifact_name
        else:
            assert isinstance(artifact, TensorRTLLMArtifactIdentity)
            authenticated_model_id = artifact.bundle_name
        if artifact_request.get("model_id") != authenticated_model_id:
            raise EvidencePackError(
                f"{side} request model_id does not match provider artifact identity"
            )
    for role, expected in expected_locators.items():
        if identities[role].locator != expected:
            raise EvidencePackError(
                f"{role} input locator does not match the normalized request binding"
            )


def publish_comparison_evidence(
    destination: Path,
    *,
    comparison_id: str,
    baseline: InputIdentity,
    subject: InputIdentity,
    dataset: InputIdentity,
    baseline_runtime: InputIdentity,
    subject_runtime: InputIdentity,
    policy: InputIdentity,
    normalized_request: Mapping[str, object],
    schedule: RuntimeBehavioralSchedule,
    policy_bytes: bytes,
    baseline_evidence: RuntimeSideEvidence,
    subject_evidence: RuntimeSideEvidence,
    signing_key_path: Path | ed25519.Ed25519PrivateKey,
    observations: Sequence[EvidenceObservation] = (),
    scorer_registry: ScorerExtensionRegistry | None = None,
    expected_paired_records: Mapping[str, object] | None = None,
) -> Path:
    """Derive, preflight, sign, and atomically publish one comparison."""

    destination = Path(destination)
    if destination.name in {"", ".", ".."}:
        raise EvidencePackError("evidence destination must name a directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise EvidencePackError("evidence destination parent must not be a symlink")
    if os.path.lexists(destination):
        raise EvidencePackError(f"evidence destination already exists: {destination}")
    request_errors = evaluation_request_errors(normalized_request)
    if request_errors:
        raise EvidencePackError(request_errors[0])
    requested_observations = normalized_request.get("observations", [])
    expected_observations = [
        {
            "id": observation.observation_id,
            "kind": observation.kind,
            "scope": observation.scope,
            "payload_digest": sha256_digest(observation.payload),
        }
        for observation in sorted(observations, key=lambda item: item.observation_id)
    ]
    if requested_observations != expected_observations:
        raise EvidencePackError(
            "normalized request observations do not match publication inputs"
        )
    metric = request_metric(normalized_request)
    scorer_binding = request_scorer_binding(normalized_request)
    schedule_payload = schedule_bytes(schedule)
    expected_dataset = f"sha256:{schedule.schedule_sha256}"
    if normalize_digest(dataset.digest, label="dataset digest") != expected_dataset:
        raise EvidencePackError(
            "dataset identity must equal the canonical evaluation schedule digest"
        )
    if sha256_digest(policy_bytes) != normalize_digest(
        policy.digest, label="policy digest"
    ):
        raise EvidencePackError("policy identity does not match policy bytes")
    policy_payload = parse_json_object(policy_bytes, label="policy")
    comparison = normalized_request["comparison"]
    assert isinstance(comparison, Mapping)

    def provider_name(side: str) -> str:
        side_request = comparison[side]
        assert isinstance(side_request, Mapping)
        runtime = side_request["runtime"]
        assert isinstance(runtime, Mapping)
        value = runtime["provider"]
        assert isinstance(value, str)
        return value

    paired = derive_paired_records(
        schedule=schedule,
        metric=metric,
        baseline=baseline_evidence,
        subject=subject_evidence,
        baseline_identity_digest=baseline.digest,
        subject_identity_digest=subject.digest,
        baseline_runtime_digest=baseline_runtime.digest,
        subject_runtime_digest=subject_runtime.digest,
        scorer_binding=scorer_binding,
        scorer_registry=scorer_registry,
    )
    if expected_paired_records is not None and paired != dict(expected_paired_records):
        raise EvidencePackError(
            "publication paired records do not match transaction-derived records"
        )
    report = build_comparison_report(
        comparison_id=comparison_id,
        paired_records=paired,
        policy=policy_payload,
        policy_digest=policy.digest,
    )

    identities = {
        "baseline": baseline,
        "subject": subject,
        "dataset": dataset,
        "baseline_runtime": baseline_runtime,
        "subject_runtime": subject_runtime,
        "policy": policy,
    }
    _validate_input_bindings(
        normalized_request,
        schedule=schedule,
        identities=identities,
        baseline_evidence=baseline_evidence,
        subject_evidence=subject_evidence,
    )
    payload_files: dict[str, bytes] = {}
    input_manifest: dict[str, object] = {}
    for role in INPUT_ROLES:
        identity = identities[role]
        payload = canonical_json_bytes(identity_payload(role, identity))
        relative = f"inputs/{role}.json"
        payload_files[relative] = payload
        input_manifest[role] = {
            "path": relative,
            "digest": sha256_digest(payload),
            "material_digest": normalize_digest(
                identity.digest, label=f"{role} digest"
            ),
        }

    payload_files[EVIDENCE_PATHS["request"]] = canonical_json_bytes(
        dict(normalized_request)
    )
    payload_files[EVIDENCE_PATHS["schedule"]] = schedule_payload
    payload_files[EVIDENCE_PATHS["evaluation_report"]] = canonical_json_bytes(report)
    payload_files.update(_side_payloads("baseline", baseline_evidence))
    payload_files.update(_side_payloads("subject", subject_evidence))
    if len(observations) > MAX_OBSERVATIONS:
        raise EvidencePackError(
            f"evidence pack supports at most {MAX_OBSERVATIONS} observations"
        )
    observation_manifest: dict[str, object] = {}
    for observation in observations:
        if observation.observation_id in observation_manifest:
            raise EvidencePackError(
                f"duplicate observation_id: {observation.observation_id!r}"
            )
        relative = f"observations/{observation.observation_id}.json"
        envelope = evidence_observation_bytes(
            observation,
            comparison_id=comparison_id,
            schedule_digest=dataset.digest,
            policy_digest=policy.digest,
            artifact_digests={
                "baseline": baseline.digest,
                "subject": subject.digest,
            },
        )
        payload_files[relative] = envelope
        observation_manifest[observation.observation_id] = {
            "path": relative,
            "digest": sha256_digest(envelope),
            "kind": observation.kind,
            "scope": observation.scope,
        }
    for relative, payload in payload_files.items():
        if not isinstance(payload, bytes) or not payload:
            raise EvidencePackError(f"{relative} must contain non-empty bytes")
        if len(payload) > MAX_EVIDENCE_BYTES:
            raise EvidencePackError(
                f"{relative} exceeds the {MAX_EVIDENCE_BYTES}-byte limit"
            )

    evidence_manifest = {
        role: {
            "path": relative,
            "digest": sha256_digest(payload_files[relative]),
        }
        for role, relative in EVIDENCE_PATHS.items()
    }
    paired_payload = canonical_json_bytes(paired)
    records_path = "records/paired-records.json"
    payload_files[records_path] = paired_payload
    records = paired["records"]
    assert isinstance(records, list)

    private_key = (
        signing_key_path
        if isinstance(signing_key_path, ed25519.Ed25519PrivateKey)
        else _load_private_key(Path(signing_key_path))
    )
    fingerprint = integrity.public_key_fingerprint(private_key.public_key())
    checksums = _checksum_bytes(payload_files)
    manifest = {
        "format": EVIDENCE_PACK_FORMAT,
        "comparison_id": comparison_id,
        "inputs": input_manifest,
        "evidence": evidence_manifest,
        "paired_records": {
            "path": records_path,
            "digest": sha256_digest(paired_payload),
            "count": len(records),
        },
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": hashlib.sha256(checksums).hexdigest(),
        "signing_key_fingerprint": fingerprint,
    }
    if observation_manifest:
        manifest["observations"] = observation_manifest
    manifest_bytes = canonical_json_bytes(manifest)
    signature_bytes, derived_fingerprint = _signature_bytes(
        manifest_bytes, private_key=private_key
    )
    assert fingerprint == derived_fingerprint

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    published = False
    try:
        for relative, payload in sorted(payload_files.items()):
            _write_new(staging / relative, payload)
        _preflight_runtime_side(
            staging,
            side="baseline",
            runtime_digest=normalize_digest(
                baseline_runtime.digest, label="baseline runtime digest"
            ),
            provider_name=provider_name("baseline"),
            schedule_sha256=schedule.schedule_sha256,
            policy_digest=policy.digest,
        )
        _preflight_runtime_side(
            staging,
            side="subject",
            runtime_digest=normalize_digest(
                subject_runtime.digest, label="subject runtime digest"
            ),
            provider_name=provider_name("subject"),
            schedule_sha256=schedule.schedule_sha256,
            policy_digest=policy.digest,
        )
        _write_new(staging / "checksums.sha256", checksums)
        _write_new(staging / "manifest.json", manifest_bytes)
        _write_new(staging / integrity.MANIFEST_SIGNATURE_FILENAME, signature_bytes)
        _publish_directory_no_clobber(staging, destination)
        published = True
        for directory in sorted(
            (path for path in destination.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        destination.chmod(0o555)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return destination


__all__ = ["publish_comparison_evidence"]
