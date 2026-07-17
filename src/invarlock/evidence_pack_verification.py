"""Independent verification for canonical InvarLock evidence packs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from invarlock import evidence_pack_integrity as integrity
from invarlock.core.runtime_provider import (
    GGUFArtifactIdentity,
    HFSnapshotArtifactIdentity,
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
    build_runtime_behavioral_schedule,
)
from invarlock.core.scorer_extension import ScorerExtensionRegistry
from invarlock.evidence_pack_contract import (
    EVIDENCE_INPUT_IDENTITY_FORMAT,
    EVIDENCE_PACK_FORMAT,
    EVIDENCE_PACK_VERIFY_FORMAT,
    EVIDENCE_PATHS,
    INPUT_ROLES,
    MAX_EVIDENCE_BYTES,
    MAX_IDENTITY_BYTES,
    EvidencePackError,
    RuntimeSideEvidence,
    build_comparison_report,
    canonical_json_bytes,
    dataset_preparation_binding_errors,
    derive_paired_records,
    evaluation_request_errors,
    evidence_observation_errors,
    normalize_digest,
    parse_json_object,
    request_metric,
    request_scorer_binding,
    runtime_side_config_errors,
    schedule_bytes,
    sha256_digest,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.evidence_pack_snapshot import PackSnapshot
from invarlock.evidence_pack_support import EvidencePackResult, EvidencePackStatus
from invarlock.public_contracts import load_evidence_pack_schema
from invarlock.runtime_provider_evidence import decode_artifact_identity
from invarlock.runtime_verify import verify_runtime_manifest_snapshot

_DIGEST_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def _load_json_object(
    path: Path, *, label: str, max_bytes: int
) -> tuple[dict[str, Any] | None, str | None, bytes | None]:
    try:
        raw = read_regular_file_bytes(path, label=label, max_bytes=max_bytes)
        payload = parse_json_bytes(raw, label=label)
    except StrictJsonError as exc:
        return None, str(exc), None
    if not isinstance(payload, dict):
        return None, f"{label} must decode to a JSON object", raw
    return payload, None, raw


def _validate_manifest(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["manifest must decode to a JSON object"]
    manifest_format = payload.get("format")
    if manifest_format != EVIDENCE_PACK_FORMAT:
        return [
            f"unsupported manifest format {manifest_format!r}; expected "
            f"{EVIDENCE_PACK_FORMAT!r}"
        ]
    schema_errors = sorted(
        Draft202012Validator(load_evidence_pack_schema()).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    errors: list[str] = []
    if schema_errors:
        first = schema_errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        errors.append(f"manifest schema failed at {location}: {first.message}")
    expected_fields = {
        "format",
        "comparison_id",
        "inputs",
        "evidence",
        "paired_records",
        "checksums_sha256",
        "checksums_sha256_digest",
        "signing_key_fingerprint",
    }
    actual_fields = frozenset(payload)
    if actual_fields not in {
        frozenset(expected_fields),
        frozenset({*expected_fields, "observations"}),
    }:
        errors.append("manifest fields are invalid")
    comparison_id = payload.get("comparison_id")
    if not isinstance(comparison_id, str) or not _IDENTIFIER_RE.fullmatch(
        comparison_id
    ):
        errors.append("manifest comparison_id is invalid")
    if payload.get("checksums_sha256") != "checksums.sha256":
        errors.append("manifest checksums_sha256 is invalid")
    checksums_digest = payload.get("checksums_sha256_digest")
    if not isinstance(checksums_digest, str) or not re.fullmatch(
        r"[a-f0-9]{64}", checksums_digest
    ):
        errors.append("manifest checksums_sha256_digest is invalid")
    fingerprint = payload.get("signing_key_fingerprint")
    if not isinstance(fingerprint, str) or not _DIGEST_RE.fullmatch(fingerprint):
        errors.append("manifest signing_key_fingerprint is invalid")

    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(INPUT_ROLES):
        errors.append(
            "manifest inputs must contain baseline, subject, dataset, both "
            "runtimes, and policy"
        )
    else:
        for role in INPUT_ROLES:
            reference = inputs.get(role)
            if not isinstance(reference, dict) or set(reference) != {
                "path",
                "digest",
                "material_digest",
            }:
                errors.append(f"manifest input {role} fields are invalid")
                continue
            if reference.get("path") != f"inputs/{role}.json":
                errors.append(f"manifest input {role} path is invalid")
            for field in ("digest", "material_digest"):
                value = reference.get(field)
                if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
                    errors.append(f"manifest input {role} {field} is invalid")

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(EVIDENCE_PATHS):
        errors.append("manifest evidence roles are invalid")
    else:
        for role, relative in EVIDENCE_PATHS.items():
            reference = evidence.get(role)
            if not isinstance(reference, dict) or set(reference) != {
                "path",
                "digest",
            }:
                errors.append(f"manifest evidence {role} fields are invalid")
                continue
            if reference.get("path") != relative:
                errors.append(f"manifest evidence {role} path is invalid")
            digest = reference.get("digest")
            if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
                errors.append(f"manifest evidence {role} digest is invalid")

    paired = payload.get("paired_records")
    if not isinstance(paired, dict) or set(paired) != {"path", "digest", "count"}:
        errors.append("manifest paired_records fields are invalid")
    else:
        if paired.get("path") != "records/paired-records.json":
            errors.append("manifest paired_records path is invalid")
        digest = paired.get("digest")
        if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            errors.append("manifest paired_records digest is invalid")
        count = paired.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            errors.append("manifest paired_records count is invalid")
    observations = payload.get("observations")
    if observations is not None:
        if not isinstance(observations, dict) or not 1 <= len(observations) <= 64:
            errors.append("manifest observations must contain between 1 and 64 entries")
        else:
            for observation_id, reference in observations.items():
                if (
                    not isinstance(observation_id, str)
                    or _IDENTIFIER_RE.fullmatch(observation_id) is None
                ):
                    errors.append("manifest observation identifier is invalid")
                    continue
                if not isinstance(reference, dict) or set(reference) != {
                    "path",
                    "digest",
                    "kind",
                    "scope",
                }:
                    errors.append(
                        f"manifest observation {observation_id!r} fields are invalid"
                    )
                    continue
                if reference.get("path") != f"observations/{observation_id}.json":
                    errors.append(
                        f"manifest observation {observation_id!r} path is invalid"
                    )
                digest = reference.get("digest")
                if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
                    errors.append(
                        f"manifest observation {observation_id!r} digest is invalid"
                    )
    return errors


def _safe_pack_path(pack_dir: Path, relative: str, *, label: str) -> Path:
    path = integrity._normalize_pack_path(pack_dir, relative)
    if path is None:
        raise EvidencePackError(f"{label} path is unsafe")
    return path


def _verify_identity(
    pack_dir: Path, *, role: str, reference: Mapping[str, object]
) -> tuple[dict[str, Any] | None, list[str]]:
    path = _safe_pack_path(pack_dir, f"inputs/{role}.json", label=role)
    payload, load_error, raw = _load_json_object(
        path, label=f"{role} identity", max_bytes=MAX_IDENTITY_BYTES
    )
    if load_error or payload is None or raw is None:
        return None, [load_error or f"{role} identity is invalid"]
    errors: list[str] = []
    required = {"format", "role", "digest"}
    if not required.issubset(payload) or set(payload) - {
        "format",
        "role",
        "digest",
        "locator",
        "media_type",
    }:
        errors.append(f"{role} identity fields are invalid")
    if payload.get("format") != EVIDENCE_INPUT_IDENTITY_FORMAT:
        errors.append(f"{role} identity format is invalid")
    if payload.get("role") != role:
        errors.append(f"{role} identity role is invalid")
    digest = payload.get("digest")
    if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
        errors.append(f"{role} identity digest is invalid")
    if canonical_json_bytes(payload) != raw:
        errors.append(f"{role} identity is not canonical JSON")
    if sha256_digest(raw) != reference.get("digest"):
        errors.append(f"{role} identity digest does not match manifest")
    if digest != reference.get("material_digest"):
        errors.append(f"{role} material digest does not match manifest")
    return payload, errors


def _request_input_binding_errors(
    request: Mapping[str, object],
    identities: Mapping[str, dict[str, Any] | None],
    loaded: Mapping[str, bytes],
) -> list[str]:
    """Bind descriptive identities to the same authenticated request roles."""

    errors: list[str] = []
    comparison = request.get("comparison")
    if not isinstance(comparison, Mapping):
        return ["normalized request comparison is invalid"]
    expected_locators: dict[str, object] = {
        "dataset": EVIDENCE_PATHS["schedule"],
        "policy": "inputs/policy.json",
    }
    for side in ("baseline", "subject"):
        side_request = comparison.get(side)
        artifact_request = (
            side_request.get("artifact") if isinstance(side_request, Mapping) else None
        )
        if not isinstance(artifact_request, Mapping):
            errors.append(f"normalized request {side} artifact is invalid")
            continue
        expected_locators[side] = artifact_request.get("locator")
        requested_model_id = artifact_request.get("model_id")
        try:
            artifact = decode_artifact_identity(loaded[f"{side}_provider_identity"])
        except (KeyError, ValueError) as exc:
            errors.append(f"{side} artifact identity could not be decoded: {exc}")
            continue
        if isinstance(artifact, HFSnapshotArtifactIdentity):
            authenticated_model_id = artifact.model_id
        elif isinstance(artifact, GGUFArtifactIdentity):
            authenticated_model_id = artifact.artifact_name
        else:
            assert isinstance(artifact, TensorRTLLMArtifactIdentity)
            authenticated_model_id = artifact.bundle_name
        if requested_model_id != authenticated_model_id:
            errors.append(
                f"{side} request model_id does not match provider artifact identity"
            )

    for side in ("baseline", "subject"):
        runtime_identity = identities.get(f"{side}_runtime")
        runtime_digest = (
            runtime_identity.get("digest")
            if isinstance(runtime_identity, Mapping)
            else None
        )
        expected_locators[f"{side}_runtime"] = (
            f"runtime:{runtime_digest}" if isinstance(runtime_digest, str) else None
        )
    for role, expected in expected_locators.items():
        identity = identities.get(role)
        observed = identity.get("locator") if isinstance(identity, Mapping) else None
        if observed != expected:
            errors.append(
                f"{role} input locator does not match the normalized request binding"
            )
    return errors


def _load_bound_evidence(
    pack_dir: Path, evidence: Mapping[str, object]
) -> tuple[dict[str, bytes], list[str]]:
    loaded: dict[str, bytes] = {}
    errors: list[str] = []
    for role, relative in EVIDENCE_PATHS.items():
        reference = evidence.get(role)
        if not isinstance(reference, Mapping):
            errors.append(f"manifest evidence {role} is missing")
            continue
        try:
            raw = read_regular_file_bytes(
                _safe_pack_path(pack_dir, relative, label=role),
                label=role.replace("_", " "),
                max_bytes=MAX_EVIDENCE_BYTES,
            )
        except (EvidencePackError, StrictJsonError) as exc:
            errors.append(str(exc))
            continue
        if sha256_digest(raw) != reference.get("digest"):
            errors.append(f"{role} digest does not match manifest")
        loaded[role] = raw
    return loaded, errors


def _runtime_side_from_loaded(
    side: str, loaded: Mapping[str, bytes]
) -> RuntimeSideEvidence:
    return RuntimeSideEvidence(
        run_report=loaded[f"{side}_run_report"],
        runtime_manifest=loaded[f"{side}_runtime_manifest"],
        runtime_config=loaded[f"{side}_runtime_config"],
        artifact_identity=loaded[f"{side}_provider_identity"],
        provider_receipt=loaded[f"{side}_provider_receipt"],
        scoring_observation=loaded[f"{side}_scoring_observation"],
    )


def _verify_runtime_side(
    pack_dir: Path,
    *,
    side: str,
    loaded: Mapping[str, bytes],
    expected_runtime_digest: str,
) -> list[str]:
    try:
        manifest = parse_json_object(
            loaded[f"{side}_runtime_manifest"], label=f"{side} runtime manifest"
        )
    except EvidencePackError as exc:
        return [str(exc)]
    report_path = pack_dir / EVIDENCE_PATHS[f"{side}_run_report"]
    manifest_path = pack_dir / EVIDENCE_PATHS[f"{side}_runtime_manifest"]
    result = verify_runtime_manifest_snapshot(
        loaded[f"{side}_run_report"],
        manifest,
        report=report_path,
        manifest=manifest_path,
        expected_image_digest=expected_runtime_digest,
        require_strict_runtime=True,
    )
    return [f"{side} runtime manifest: {error}" for error in result.errors]


def _verify_observations(
    pack_dir: Path,
    *,
    references: object,
    requested: object,
    comparison_id: str,
    schedule_digest: str,
    policy_digest: str,
    artifact_digests: Mapping[str, str],
) -> tuple[list[dict[str, str]], list[str]]:
    requested_items = [] if requested is None else requested
    if not isinstance(requested_items, list):
        return [], ["normalized request observations are invalid"]
    requested_by_id: dict[str, Mapping[str, object]] = {}
    for item in requested_items:
        observation_id = item.get("id") if isinstance(item, Mapping) else None
        if (
            not isinstance(observation_id, str)
            or observation_id in requested_by_id
            or not isinstance(item, Mapping)
        ):
            return [], ["normalized request observation entry is invalid"]
        requested_by_id[observation_id] = item
    if references is None:
        return (
            ([], [])
            if not requested_by_id
            else ([], ["normalized request observations are missing from manifest"])
        )
    if not isinstance(references, Mapping):
        return [], ["manifest observations are invalid"]
    if set(references) != set(requested_by_id):
        return [], ["manifest observations do not match normalized request"]
    verified: list[dict[str, str]] = []
    errors: list[str] = []
    for observation_id, reference in sorted(references.items()):
        if not isinstance(observation_id, str) or not isinstance(reference, Mapping):
            errors.append("manifest observation entry is invalid")
            continue
        relative = reference.get("path")
        if not isinstance(relative, str):
            errors.append(f"observation {observation_id!r} path is invalid")
            continue
        try:
            raw = read_regular_file_bytes(
                _safe_pack_path(
                    pack_dir,
                    relative,
                    label=f"observation {observation_id}",
                ),
                label=f"observation {observation_id}",
                max_bytes=MAX_EVIDENCE_BYTES,
            )
            envelope = parse_json_object(
                raw,
                label=f"observation {observation_id}",
            )
        except (EvidencePackError, StrictJsonError) as exc:
            errors.append(str(exc))
            continue
        local_errors: list[str] = []
        if canonical_json_bytes(envelope) != raw:
            local_errors.append(
                f"observation {observation_id!r} must use canonical JSON"
            )
        digest = sha256_digest(raw)
        if digest != reference.get("digest"):
            local_errors.append(
                f"observation {observation_id!r} digest does not match manifest"
            )
        descriptor = requested_by_id[observation_id]
        observation_payload = envelope.get("payload")
        canonical_observation_payload = (
            canonical_json_bytes(observation_payload)
            if isinstance(observation_payload, Mapping)
            else None
        )
        payload_digest = (
            sha256_digest(canonical_observation_payload)
            if canonical_observation_payload is not None
            else None
        )
        expected_descriptor = {
            "id": observation_id,
            "kind": envelope.get("kind"),
            "scope": envelope.get("scope"),
            "payload_digest": payload_digest,
        }
        if dict(descriptor) != expected_descriptor:
            local_errors.append(
                f"observation {observation_id!r} does not match normalized request"
            )
        local_errors.extend(
            evidence_observation_errors(
                envelope,
                observation_id=observation_id,
                reference=reference,
                comparison_id=comparison_id,
                schedule_digest=schedule_digest,
                policy_digest=policy_digest,
                artifact_digests=artifact_digests,
            )
        )
        errors.extend(local_errors)
        if not local_errors:
            kind = envelope.get("kind")
            scope = envelope.get("scope")
            assert isinstance(kind, str) and isinstance(scope, str)
            verified.append(
                {
                    "observation_id": observation_id,
                    "kind": kind,
                    "scope": scope,
                    "digest": digest,
                }
            )
    return verified, errors


def _result(
    pack_dir: Path,
    *,
    errors: list[str],
    signer_fingerprint: str | None,
    comparison_id: str | None,
    request_digest: str | None,
    anchors: Mapping[str, object],
    status: EvidencePackStatus,
    policy_verdict: str | None = None,
    observations: tuple[dict[str, str], ...] = (),
) -> EvidencePackResult:
    integrity_ok = not errors
    ok = integrity_ok and policy_verdict != "fail"
    authenticity = (
        "pinned"
        if signer_fingerprint is not None
        and signer_fingerprint == anchors.get("signer_fingerprint")
        else "mismatch"
    )
    return EvidencePackResult(
        payload={
            "format_version": EVIDENCE_PACK_VERIFY_FORMAT,
            "pack": pack_dir.name,
            "pack_format": EVIDENCE_PACK_FORMAT,
            "comparison_id": comparison_id,
            "request_digest": request_digest,
            "ok": ok,
            "integrity_ok": integrity_ok,
            "reports_verified": integrity_ok,
            "verification_scope": (
                "paired_comparison" if integrity_ok else "not_verified"
            ),
            "assurance_status": "verified" if integrity_ok else "not_verified",
            "policy_verdict": policy_verdict,
            "observations": list(observations),
            "authenticity": authenticity,
            "signer_fingerprint": signer_fingerprint,
            "anchors": dict(anchors),
            "warnings": [],
            "errors": errors,
        },
        status=(
            EvidencePackStatus.OK
            if ok
            else (
                EvidencePackStatus.REPORTS
                if integrity_ok and policy_verdict == "fail"
                else status
            )
        ),
    )


def _snapshot_failure_result(
    pack_dir: Path,
    *,
    errors: list[str],
    policy_path: Path | None,
    expected_artifact_digests: Mapping[str, str] | None,
    expected_schedule_digest: str | None,
    expected_runtime_digests: Mapping[str, str] | None,
    expected_signer_fingerprint: str | None,
    manifest_digest: str | None = None,
) -> EvidencePackResult:
    """Return a closed failure when immutable snapshot handling fails."""

    anchors: dict[str, object] = {
        "policy_digest": None,
        "artifact_digests": (
            dict(expected_artifact_digests)
            if isinstance(expected_artifact_digests, Mapping)
            else {}
        ),
        "schedule_digest": expected_schedule_digest,
        "runtime_digests": (
            dict(expected_runtime_digests)
            if isinstance(expected_runtime_digests, Mapping)
            else {}
        ),
        "signer_fingerprint": integrity.normalize_expected_fingerprint(
            expected_signer_fingerprint
        ),
    }
    result = _result(
        pack_dir,
        errors=errors,
        signer_fingerprint=None,
        comparison_id=None,
        request_digest=None,
        anchors=anchors,
        status=EvidencePackStatus.INTEGRITY,
    )
    return EvidencePackResult(result.payload, result.status, manifest_digest)


def _with_snapshot_errors(
    result: EvidencePackResult,
    *,
    pack_dir: Path,
    errors: list[str],
    manifest_digest: str | None,
) -> EvidencePackResult:
    payload = dict(result.payload)
    existing = payload.get("errors")
    observed = list(existing) if isinstance(existing, list) else []
    observed.extend(errors)
    payload.update(
        {
            "pack": pack_dir.name,
            "ok": False,
            "integrity_ok": False,
            "reports_verified": False,
            "verification_scope": "not_verified",
            "assurance_status": "not_verified",
            "errors": list(dict.fromkeys(str(error) for error in observed)),
        }
    )
    return EvidencePackResult(
        payload,
        EvidencePackStatus.INTEGRITY,
        manifest_digest,
    )


def _verify_comparison_evidence_snapshot(
    pack_dir: Path,
    *,
    policy_path: Path | None,
    policy_bytes: bytes | None,
    expected_artifact_digests: Mapping[str, str] | None,
    expected_schedule_digest: str | None,
    expected_runtime_digests: Mapping[str, str] | None,
    expected_signer_fingerprint: str | None,
    scorer_registry: ScorerExtensionRegistry | None,
) -> EvidencePackResult:
    """Verify one already materialized immutable bundle snapshot."""

    pack_dir = Path(pack_dir)
    errors: list[str] = []
    policy_payload: dict[str, Any] | None = None
    policy_digest: str | None = None
    if policy_bytes is not None:
        if len(policy_bytes) > 4 * 1024 * 1024:
            errors.append(
                "independent policy anchor exceeds the 4194304-byte size limit"
            )
        else:
            try:
                policy_payload = parse_json_object(
                    policy_bytes, label="independent policy anchor"
                )
                policy_digest = sha256_digest(policy_bytes)
            except (EvidencePackError, StrictJsonError) as exc:
                errors.append(str(exc))
    elif policy_path is None:
        errors.append("independent policy_path anchor is required")
    else:
        try:
            loaded_policy_bytes = read_regular_file_bytes(
                Path(policy_path),
                label="independent policy anchor",
                max_bytes=4 * 1024 * 1024,
            )
            policy_payload = parse_json_object(
                loaded_policy_bytes, label="independent policy anchor"
            )
            policy_digest = sha256_digest(loaded_policy_bytes)
        except (EvidencePackError, StrictJsonError) as exc:
            errors.append(str(exc))
    artifacts: dict[str, str] = {}
    if not isinstance(expected_artifact_digests, Mapping) or set(
        expected_artifact_digests
    ) != {"baseline", "subject"}:
        errors.append(
            "independent artifact anchors must contain exactly baseline and subject"
        )
    else:
        for side in ("baseline", "subject"):
            try:
                artifacts[side] = normalize_digest(
                    expected_artifact_digests[side],
                    label=f"independent {side} artifact anchor",
                )
            except EvidencePackError as exc:
                errors.append(str(exc))
    try:
        schedule_digest = normalize_digest(
            expected_schedule_digest or "",
            label="independent schedule anchor",
        )
    except EvidencePackError as exc:
        errors.append(str(exc))
        schedule_digest = None
    runtimes: dict[str, str] = {}
    if not isinstance(expected_runtime_digests, Mapping) or set(
        expected_runtime_digests
    ) != {"baseline", "subject"}:
        errors.append(
            "independent runtime anchors must contain exactly baseline and subject"
        )
    else:
        for side in ("baseline", "subject"):
            try:
                runtimes[side] = normalize_digest(
                    expected_runtime_digests[side],
                    label=f"independent {side} runtime anchor",
                )
            except EvidencePackError as exc:
                errors.append(str(exc))
    normalized_signer = integrity.normalize_expected_fingerprint(
        expected_signer_fingerprint
    )
    if normalized_signer is None:
        errors.append("independent signer anchor must be a sha256:... fingerprint")
    anchors: dict[str, object] = {
        "policy_digest": policy_digest,
        "artifact_digests": artifacts,
        "schedule_digest": schedule_digest,
        "runtime_digests": runtimes,
        "signer_fingerprint": normalized_signer,
    }

    manifest, manifest_error, manifest_raw = _load_json_object(
        pack_dir / "manifest.json", label="manifest.json", max_bytes=256 * 1024
    )
    if manifest_error or manifest is None or manifest_raw is None:
        errors.append(manifest_error or "manifest.json is invalid")
        return _result(
            pack_dir,
            errors=errors,
            signer_fingerprint=None,
            comparison_id=None,
            request_digest=None,
            anchors=anchors,
            status=EvidencePackStatus.FORMAT,
        )
    format_errors = _validate_manifest(manifest)
    if canonical_json_bytes(manifest) != manifest_raw:
        format_errors.append("manifest.json is not canonical JSON")
    errors.extend(format_errors)
    signer_fingerprint: str | None = None
    policy_verdict: str | None = None
    request_digest: str | None = None
    verified_observations: list[dict[str, str]] = []
    if not format_errors:
        signature_errors, _warnings, signer_fingerprint = integrity.verify_signature(
            pack_dir,
            strict=True,
            expected_fingerprints=(
                frozenset({normalized_signer}) if normalized_signer else None
            ),
        )
        errors.extend(signature_errors)
        try:
            checksums = read_regular_file_bytes(
                pack_dir / "checksums.sha256",
                label="checksums.sha256",
                max_bytes=1024 * 1024,
            )
        except StrictJsonError as exc:
            errors.append(str(exc))
            checksums = b""
        errors.extend(
            integrity.verify_manifest_binds_checksums_payload(manifest, checksums)
        )
        checksum_errors, covered = integrity.verify_checksums(pack_dir)
        errors.extend(checksum_errors)
        extra_errors, _warnings = integrity.verify_no_extra_files(
            pack_dir, covered_paths=covered, strict=True
        )
        errors.extend(extra_errors)

        identities: dict[str, dict[str, Any] | None] = {}
        inputs = manifest["inputs"]
        assert isinstance(inputs, dict)
        for role in INPUT_ROLES:
            reference = inputs[role]
            assert isinstance(reference, dict)
            identity, identity_errors = _verify_identity(
                pack_dir, role=role, reference=reference
            )
            identities[role] = identity
            errors.extend(identity_errors)

        evidence = manifest["evidence"]
        assert isinstance(evidence, dict)
        loaded, evidence_errors = _load_bound_evidence(pack_dir, evidence)
        errors.extend(evidence_errors)
        required_loaded = set(EVIDENCE_PATHS)
        if required_loaded.issubset(loaded):
            try:
                request = parse_json_object(
                    loaded["request"], label="normalized request"
                )
                if canonical_json_bytes(request) != loaded["request"]:
                    errors.append("normalized request is not canonical JSON")
                request_digest = sha256_digest(loaded["request"])
                errors.extend(evaluation_request_errors(request))
                metric = request_metric(request)
                scorer_binding = request_scorer_binding(request)
                schedule_payload = parse_json_object(
                    loaded["schedule"], label="runtime behavioral schedule"
                )
                schedule = build_runtime_behavioral_schedule(schedule_payload)
                if schedule_bytes(schedule) != loaded["schedule"]:
                    errors.append("runtime behavioral schedule is not canonical JSON")
                errors.extend(dataset_preparation_binding_errors(request, schedule))
                identity_digests = {
                    role: (
                        identity.get("digest")
                        if isinstance(identity, Mapping)
                        else None
                    )
                    for role, identity in identities.items()
                }
                embedded_artifacts = {
                    side: digest
                    for side in ("baseline", "subject")
                    if isinstance((digest := identity_digests.get(side)), str)
                }
                embedded_policy = identity_digests.get("policy")
                embedded_schedule = identity_digests.get("dataset")
                if (
                    len(embedded_artifacts) == 2
                    and isinstance(embedded_policy, str)
                    and isinstance(embedded_schedule, str)
                ):
                    verified_observations, observation_errors = _verify_observations(
                        pack_dir,
                        references=manifest.get("observations"),
                        requested=request.get("observations"),
                        comparison_id=manifest["comparison_id"],
                        schedule_digest=embedded_schedule,
                        policy_digest=embedded_policy,
                        artifact_digests=embedded_artifacts,
                    )
                    errors.extend(observation_errors)
                for side in ("baseline", "subject"):
                    if (
                        artifacts.get(side) is not None
                        and identity_digests.get(side) != artifacts[side]
                    ):
                        errors.append(
                            f"embedded {side} artifact identity does not match caller "
                            "artifact anchor"
                        )
                if identity_digests.get("dataset") != (
                    f"sha256:{schedule.schedule_sha256}"
                ):
                    errors.append(
                        "dataset identity does not match the canonical schedule"
                    )
                if (
                    schedule_digest is not None
                    and identity_digests.get("dataset") != schedule_digest
                ):
                    errors.append(
                        "embedded canonical schedule identity does not match caller "
                        "schedule anchor"
                    )
                if (
                    policy_digest is not None
                    and identity_digests.get("policy") != policy_digest
                ):
                    errors.append(
                        "embedded policy identity does not match caller policy anchor"
                    )
                for side in ("baseline", "subject"):
                    if (
                        runtimes.get(side) is not None
                        and identity_digests.get(f"{side}_runtime") != runtimes[side]
                    ):
                        errors.append(
                            f"embedded {side} runtime identity does not match caller "
                            "runtime anchor"
                        )
                errors.extend(
                    _request_input_binding_errors(request, identities, loaded)
                )
                comparison = request.get("comparison")
                if isinstance(comparison, Mapping) and policy_digest is not None:
                    for side in ("baseline", "subject"):
                        side_request = comparison.get(side)
                        runtime_request = (
                            side_request.get("runtime")
                            if isinstance(side_request, Mapping)
                            else None
                        )
                        provider_name = (
                            runtime_request.get("provider")
                            if isinstance(runtime_request, Mapping)
                            else None
                        )
                        if not isinstance(provider_name, str):
                            errors.append(
                                f"normalized request {side} provider is invalid"
                            )
                            continue
                        try:
                            artifact = decode_artifact_identity(
                                loaded[f"{side}_provider_identity"]
                            )
                        except ValueError as exc:
                            errors.append(str(exc))
                            continue
                        errors.extend(
                            runtime_side_config_errors(
                                loaded[f"{side}_runtime_config"],
                                role=side,
                                provider_name=provider_name,
                                artifact_identity_sha256=(
                                    artifact_identity_sha256(artifact)
                                ),
                                schedule_sha256=schedule.schedule_sha256,
                                policy_digest=policy_digest,
                            )
                        )
                if not errors:
                    baseline_side = _runtime_side_from_loaded("baseline", loaded)
                    subject_side = _runtime_side_from_loaded("subject", loaded)
                    derived = derive_paired_records(
                        schedule=schedule,
                        metric=metric,
                        baseline=baseline_side,
                        subject=subject_side,
                        baseline_identity_digest=(
                            identity_digests.get("baseline") or ""
                        ),
                        subject_identity_digest=(identity_digests.get("subject") or ""),
                        baseline_runtime_digest=runtimes["baseline"],
                        subject_runtime_digest=runtimes["subject"],
                        scorer_binding=scorer_binding,
                        scorer_registry=scorer_registry,
                    )
                    errors.extend(
                        _verify_runtime_side(
                            pack_dir,
                            side="baseline",
                            loaded=loaded,
                            expected_runtime_digest=runtimes["baseline"],
                        )
                    )
                    errors.extend(
                        _verify_runtime_side(
                            pack_dir,
                            side="subject",
                            loaded=loaded,
                            expected_runtime_digest=runtimes["subject"],
                        )
                    )
                    paired_reference = manifest["paired_records"]
                    assert isinstance(paired_reference, dict)
                    paired_path = _safe_pack_path(
                        pack_dir, "records/paired-records.json", label="paired records"
                    )
                    paired_raw = read_regular_file_bytes(
                        paired_path,
                        label="paired records",
                        max_bytes=MAX_EVIDENCE_BYTES,
                    )
                    stored_paired = parse_json_object(
                        paired_raw, label="paired records"
                    )
                    if canonical_json_bytes(stored_paired) != paired_raw:
                        errors.append("paired records are not canonical JSON")
                    if sha256_digest(paired_raw) != paired_reference.get("digest"):
                        errors.append("paired records digest does not match manifest")
                    records = stored_paired.get("records")
                    if not isinstance(records, list) or len(
                        records
                    ) != paired_reference.get("count"):
                        errors.append("paired records count does not match manifest")
                    if stored_paired != derived:
                        errors.append(
                            "paired records do not match scores derived from provider "
                            "observations"
                        )
                    if policy_payload is not None and policy_digest is not None:
                        expected_report = build_comparison_report(
                            comparison_id=manifest["comparison_id"],
                            paired_records=derived,
                            policy=policy_payload,
                            policy_digest=policy_digest,
                        )
                        report = parse_json_object(
                            loaded["evaluation_report"], label="comparison report"
                        )
                        if canonical_json_bytes(report) != loaded["evaluation_report"]:
                            errors.append("comparison report is not canonical JSON")
                        if report != expected_report:
                            errors.append(
                                "comparison report does not match verifier replay"
                            )
                        else:
                            verdict = report.get("verdict")
                            if verdict in {"pass", "fail"}:
                                policy_verdict = verdict
            except (EvidencePackError, StrictJsonError, ValueError) as exc:
                errors.append(str(exc))

    return _result(
        pack_dir,
        errors=errors,
        signer_fingerprint=signer_fingerprint,
        comparison_id=(
            manifest.get("comparison_id")
            if isinstance(manifest.get("comparison_id"), str)
            else None
        ),
        request_digest=request_digest,
        anchors=anchors,
        status=(
            EvidencePackStatus.FORMAT if format_errors else EvidencePackStatus.INTEGRITY
        ),
        policy_verdict=policy_verdict,
        observations=tuple(verified_observations),
    )


def verify_comparison_evidence(
    pack_dir: Path,
    *,
    policy_path: Path | None,
    expected_artifact_digests: Mapping[str, str] | None,
    expected_schedule_digest: str | None,
    expected_runtime_digests: Mapping[str, str] | None,
    expected_signer_fingerprint: str | None,
    scorer_registry: ScorerExtensionRegistry | None = None,
    policy_bytes: bytes | None = None,
) -> EvidencePackResult:
    """Verify one immutable pack snapshot against caller-owned trust roots.

    ``policy_bytes`` is an already authenticated snapshot used by trust-profile
    mode. Explicit path mode retains its existing ``policy_path`` behavior.
    """

    source = Path(pack_dir)
    snapshot, capture_errors = PackSnapshot.capture(
        source,
        validate_structural_json=False,
    )
    if snapshot is None:
        return _snapshot_failure_result(
            source,
            errors=capture_errors,
            policy_path=policy_path,
            expected_artifact_digests=expected_artifact_digests,
            expected_schedule_digest=expected_schedule_digest,
            expected_runtime_digests=expected_runtime_digests,
            expected_signer_fingerprint=expected_signer_fingerprint,
        )

    manifest_entry = snapshot.files.entry("manifest.json")
    manifest_digest = (
        f"sha256:{manifest_entry.sha256}" if manifest_entry is not None else None
    )
    materialized_errors: list[str] = []
    try:
        with snapshot.files.materialized() as snapshot_root:
            result = _verify_comparison_evidence_snapshot(
                snapshot_root,
                policy_path=policy_path,
                policy_bytes=policy_bytes,
                expected_artifact_digests=expected_artifact_digests,
                expected_schedule_digest=expected_schedule_digest,
                expected_runtime_digests=expected_runtime_digests,
                expected_signer_fingerprint=expected_signer_fingerprint,
                scorer_registry=scorer_registry,
            )
            materialized_errors = snapshot.files.materialized_stability_errors(
                snapshot_root
            )
    except RuntimeError as exc:
        return _snapshot_failure_result(
            source,
            errors=[str(exc)],
            policy_path=policy_path,
            expected_artifact_digests=expected_artifact_digests,
            expected_schedule_digest=expected_schedule_digest,
            expected_runtime_digests=expected_runtime_digests,
            expected_signer_fingerprint=expected_signer_fingerprint,
            manifest_digest=manifest_digest,
        )

    stability_errors = [*materialized_errors, *snapshot.stability_errors()]
    if stability_errors:
        return _with_snapshot_errors(
            result,
            pack_dir=source,
            errors=stability_errors,
            manifest_digest=manifest_digest,
        )
    payload = dict(result.payload)
    payload["pack"] = source.name
    return EvidencePackResult(payload, result.status, manifest_digest)


__all__ = ["verify_comparison_evidence"]
