"""Portable in-toto/DSSE transport for one InvarLock technical decision.

The embedded signed receipt remains the authoritative replayable result.  This
module authenticates and transports that result; it does not replace evidence
replay or make a recipient's deployment decision.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.evidence_receipt import (
    SIGNED_RECEIPT_FORMAT_V1,
    SIGNED_RECEIPT_FORMAT_V2,
    SIGNED_RECEIPT_SIGNATURE_FORMAT,
)
from invarlock.public_contracts import (
    ACCEPTANCE_PREDICATE_FORMAT_VERSION,
    RECIPIENT_ACCEPTANCE_POLICY_FORMAT_VERSION,
    load_acceptance_predicate_schema,
    load_model_artifact_identity_schema,
    load_recipient_acceptance_policy_schema,
)
from invarlock.runtime_providers.tensorrt_llm_identity import (
    read_tensorrt_llm_engine_tree_sha256,
)

ACCEPTANCE_PREDICATE_TYPE = "https://invarlock.dev/attestations/acceptance/v2"
ACCEPTANCE_PREDICATE_FORMAT = ACCEPTANCE_PREDICATE_FORMAT_VERSION
RECIPIENT_POLICY_FORMAT = RECIPIENT_ACCEPTANCE_POLICY_FORMAT_VERSION
IN_TOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
DSSE_PAYLOAD_TYPE = "application/vnd.in-toto+json"
RECEIPT_MEDIA_TYPE = "application/vnd.invarlock.verification-receipt+json"

_DIGEST_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")
_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_SUPPORTED_V013_RECEIPTS = {
    SIGNED_RECEIPT_FORMAT_V1,
    SIGNED_RECEIPT_FORMAT_V2,
}


class AcceptanceAttestationError(ValueError):
    """Raised when an attestation cannot be safely created."""


@dataclass(frozen=True)
class AcceptanceAttestation:
    """A newly written canonical DSSE envelope."""

    path: Path
    statement: dict[str, Any]
    signer_fingerprint: str


@dataclass(frozen=True)
class AcceptanceDecision:
    """Recipient-side authentication and current-policy result."""

    envelope_authenticated: bool
    receipt_authenticated: bool
    subject_bound: bool
    historical_technical_verdict: str | None
    accepted: bool
    statement: dict[str, Any] | None
    errors: tuple[str, ...]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AcceptanceAttestationError(
            f"attestation value is not canonical JSON: {exc}"
        ) from exc


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _normalized_digest(value: object, *, label: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if _DIGEST_RE.fullmatch(normalized) is None:
        raise AcceptanceAttestationError(f"{label} must be a sha256:... digest")
    return normalized


def _safe_identity(value: object, *, label: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _IDENTITY_RE.fullmatch(normalized) is None:
        raise AcceptanceAttestationError(f"{label} is invalid")
    return normalized


def _load_object(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = read_regular_file_bytes(path, label=label, max_bytes=_MAX_JSON_BYTES)
        value = parse_json_bytes(raw, label=label)
    except StrictJsonError as exc:
        raise AcceptanceAttestationError(str(exc)) from exc
    if not isinstance(value, dict):
        raise AcceptanceAttestationError(f"{label} must be a JSON object")
    return raw, value


def _load_private_key(path: Path) -> ed25519.Ed25519PrivateKey:
    try:
        raw = read_regular_file_bytes(
            path, max_bytes=64 * 1024, label="attestation signing key"
        )
        key = serialization.load_pem_private_key(raw, password=None)
    except (StrictJsonError, TypeError, ValueError) as exc:
        raise AcceptanceAttestationError(
            f"could not load attestation signing key: {exc}"
        ) from exc
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise AcceptanceAttestationError("attestation signing key must be Ed25519")
    return key


def _load_public_key(
    value: Path | bytes | ed25519.Ed25519PublicKey,
) -> ed25519.Ed25519PublicKey:
    if isinstance(value, ed25519.Ed25519PublicKey):
        return value
    try:
        raw = (
            read_regular_file_bytes(
                value, max_bytes=64 * 1024, label="recipient public key"
            )
            if isinstance(value, Path)
            else value
        )
        key = serialization.load_pem_public_key(raw)
    except (StrictJsonError, TypeError, ValueError) as exc:
        raise AcceptanceAttestationError(
            f"could not load recipient public key: {exc}"
        ) from exc
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise AcceptanceAttestationError("recipient public key must be Ed25519")
    return key


def _timestamp(value: datetime | None, *, label: str) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise AcceptanceAttestationError(f"{label} must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise AcceptanceAttestationError(f"{label} is invalid")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise AcceptanceAttestationError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AcceptanceAttestationError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def _dsse_pae(payload_type: str, payload: bytes) -> bytes:
    type_bytes = payload_type.encode("utf-8")
    return (
        b"DSSEv1 "
        + str(len(type_bytes)).encode("ascii")
        + b" "
        + type_bytes
        + b" "
        + str(len(payload)).encode("ascii")
        + b" "
        + payload
    )


def _write_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    except FileExistsError as exc:
        raise AcceptanceAttestationError(
            f"attestation destination already exists: {path.name}"
        ) from exc
    except OSError as exc:
        if created:
            path.unlink(missing_ok=True)
        raise AcceptanceAttestationError(
            "could not write acceptance attestation"
        ) from exc


def _receipt_public_key(signature: object) -> ed25519.Ed25519PublicKey:
    if not isinstance(signature, dict):
        raise AcceptanceAttestationError("receipt signature is invalid")
    if set(signature) != {"format", "algorithm", "public_key", "value"}:
        raise AcceptanceAttestationError("receipt signature fields are invalid")
    if signature.get("format") != SIGNED_RECEIPT_SIGNATURE_FORMAT:
        raise AcceptanceAttestationError("receipt signature format is invalid")
    if signature.get("algorithm") != "ed25519":
        raise AcceptanceAttestationError("receipt signature algorithm is invalid")
    public_block = signature.get("public_key")
    public_value = (
        public_block.get("value")
        if isinstance(public_block, dict) and public_block.get("encoding") == "pem"
        else None
    )
    try:
        if not isinstance(public_value, str):
            raise ValueError("missing PEM value")
        loaded = serialization.load_pem_public_key(public_value.encode("ascii"))
    except (TypeError, ValueError) as exc:
        raise AcceptanceAttestationError(
            f"receipt public key is invalid: {exc}"
        ) from exc
    if not isinstance(loaded, ed25519.Ed25519PublicKey):
        raise AcceptanceAttestationError("receipt public key must be Ed25519")
    return loaded


def _authenticate_receipt(
    receipt: object,
) -> tuple[dict[str, Any], str, str]:
    if not isinstance(receipt, dict) or set(receipt) != {"statement", "signature"}:
        raise AcceptanceAttestationError("embedded receipt fields are invalid")
    statement = receipt.get("statement")
    signature = receipt.get("signature")
    if not isinstance(statement, dict):
        raise AcceptanceAttestationError("embedded receipt statement is invalid")
    if set(statement) != {
        "format",
        "pack_manifest_digest",
        "anchors",
        "verifier",
        "verdict",
    }:
        raise AcceptanceAttestationError(
            "embedded receipt statement fields are invalid"
        )
    if statement.get("format") not in _SUPPORTED_V013_RECEIPTS:
        raise AcceptanceAttestationError("embedded receipt format is unsupported")
    verifier = statement.get("verifier")
    if not isinstance(verifier, dict) or set(verifier) != {
        "identity",
        "signing_key_fingerprint",
        "trust_profile_digest",
    }:
        raise AcceptanceAttestationError("embedded receipt verifier is invalid")
    identity = _safe_identity(
        verifier.get("identity"), label="embedded receipt verifier identity"
    )
    recorded_fingerprint = _normalized_digest(
        verifier.get("signing_key_fingerprint"),
        label="embedded receipt verifier fingerprint",
    )
    trust_profile_digest = verifier.get("trust_profile_digest")
    if trust_profile_digest is not None:
        _normalized_digest(
            trust_profile_digest,
            label="embedded receipt trust-profile digest",
        )
    verdict = statement.get("verdict")
    if not isinstance(verdict, dict) or set(verdict) != {
        "ok",
        "integrity_ok",
        "policy_verdict",
        "verification_status",
    }:
        raise AcceptanceAttestationError("embedded receipt verdict is invalid")
    status = verdict.get("verification_status")
    if (
        not isinstance(verdict.get("ok"), bool)
        or not isinstance(verdict.get("integrity_ok"), bool)
        or verdict.get("policy_verdict") not in {"pass", "fail", None}
        or isinstance(status, bool)
        or not isinstance(status, int)
        or status < 0
    ):
        raise AcceptanceAttestationError("embedded receipt verdict fields are invalid")
    if verdict.get("ok") is True and (
        verdict.get("integrity_ok") is not True
        or verdict.get("policy_verdict") != "pass"
        or status != 0
    ):
        raise AcceptanceAttestationError(
            "embedded receipt successful verdict is inconsistent"
        )
    public_key = _receipt_public_key(signature)
    derived_fingerprint = public_key_fingerprint(public_key)
    if derived_fingerprint != recorded_fingerprint:
        raise AcceptanceAttestationError(
            "embedded receipt fingerprint does not match its public key"
        )
    encoded_signature = signature.get("value") if isinstance(signature, dict) else None
    try:
        if not isinstance(encoded_signature, str):
            raise ValueError("signature value is invalid")
        signature_bytes = base64.b64decode(encoded_signature, validate=True)
        public_key.verify(signature_bytes, _canonical_json_bytes(statement))
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise AcceptanceAttestationError(
            "embedded receipt signature verification failed"
        ) from exc
    return statement, identity, derived_fingerprint


def _bound_object(
    evidence: Path,
    reference: object,
    *,
    label: str,
) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(reference, dict) or set(reference) != {"path", "digest"}:
        raise AcceptanceAttestationError(f"{label} reference is invalid")
    relative = reference.get("path")
    if (
        not isinstance(relative, str)
        or relative.startswith("/")
        or ".." in Path(relative).parts
    ):
        raise AcceptanceAttestationError(f"{label} path is invalid")
    raw, value = _load_object(evidence / relative, label=label)
    if reference.get("digest") != _digest(raw):
        raise AcceptanceAttestationError(f"{label} digest does not match manifest")
    return raw, value


def _artifact(
    identity_raw: bytes,
    identity: dict[str, Any],
    *,
    anchor: object,
) -> dict[str, Any]:
    identity_digest = _digest(identity_raw)
    if identity_digest != anchor:
        raise AcceptanceAttestationError(
            "artifact identity does not match receipt anchor"
        )
    try:
        jsonschema.Draft202012Validator(load_model_artifact_identity_schema()).validate(
            identity
        )
    except jsonschema.ValidationError as exc:
        raise AcceptanceAttestationError(
            f"artifact identity is invalid: {exc.message}"
        ) from exc
    artifact_format = identity.get("artifact_format")
    if artifact_format == "hf_snapshot":
        exact = identity.get("checkpoint_tree_sha256")
        name = identity.get("model_id")
        digest_kind = "hf_snapshot_tree_sha256"
    elif artifact_format == "gguf":
        exact = identity.get("sha256")
        name = identity.get("artifact_name")
        digest_kind = "file_sha256"
    elif artifact_format == "tensorrt_llm_engine":
        exact = identity.get("engine_bundle_tree_sha256")
        name = identity.get("bundle_name")
        digest_kind = "tensorrt_llm_engine_tree_sha256"
    else:  # pragma: no cover - schema closes this branch
        raise AcceptanceAttestationError("artifact format is unsupported")
    if not isinstance(exact, str) or re.fullmatch(r"[a-f0-9]{64}", exact) is None:
        raise AcceptanceAttestationError(
            "artifact identity lacks an exact content digest"
        )
    if not isinstance(name, str) or not name:
        raise AcceptanceAttestationError("artifact identity name is invalid")
    return {
        "name": name,
        "artifact_format": artifact_format,
        "artifact_digest": "sha256:" + exact,
        "digest_kind": digest_kind,
        "artifact_identity_digest": identity_digest,
        "artifact_identity": identity,
        "artifact_identity_payload": base64.b64encode(identity_raw).decode("ascii"),
    }


def _metric(request: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    comparison = request.get("comparison")
    if not isinstance(comparison, dict):
        raise AcceptanceAttestationError("evidence request comparison is invalid")
    report_metric = report.get("metric")
    if not isinstance(report_metric, str):
        raise AcceptanceAttestationError("comparison report metric is invalid")
    scorer = comparison.get("scorer_extension")
    if scorer is None:
        if comparison.get("metric") != report_metric:
            raise AcceptanceAttestationError(
                "request metric disagrees with comparison report"
            )
        return {"kind": "built_in_metric", "name": report_metric, "scorer": None}
    if not isinstance(scorer, dict):
        raise AcceptanceAttestationError("scorer extension binding is invalid")
    required = (
        "scorer_id",
        "scorer_version",
        "descriptor_sha256",
        "configuration_sha256",
    )
    if any(not isinstance(scorer.get(field), str) for field in required):
        raise AcceptanceAttestationError("scorer extension identity is incomplete")
    return {
        "kind": "scorer_extension",
        "name": report_metric,
        "scorer": {
            field: (
                "sha256:" + scorer[field]
                if field.endswith("_sha256")
                and not str(scorer[field]).startswith("sha256:")
                else scorer[field]
            )
            for field in required
        },
    }


def _contract_release(
    receipt_format: object,
    evidence_format: object,
    report_format: object,
) -> str:
    if (
        receipt_format in _SUPPORTED_V013_RECEIPTS
        and evidence_format == "invarlock/evidence-pack-v1"
        and report_format
        in {
            "invarlock/comparison-report-v1",
            "invarlock/comparison-report-v2",
        }
    ):
        return "0.13.0"
    raise AcceptanceAttestationError(
        "receipt, evidence, and report versions are not a supported contract set"
    )


def write_acceptance_attestation(
    receipt_path: Path,
    evidence_path: Path,
    output_path: Path,
    *,
    signing_key_path: Path,
    signer_identity: str,
    policy_identity: str,
    issued_at: datetime,
    evaluation_completed_at: datetime | None = None,
) -> AcceptanceAttestation:
    """Wrap one signed receipt in a canonical in-toto Statement and DSSE envelope."""

    evidence = Path(evidence_path)
    if not evidence.is_dir() or evidence.is_symlink():
        raise AcceptanceAttestationError("evidence must be a real directory")
    receipt_raw, receipt = _load_object(
        Path(receipt_path), label="verification receipt"
    )
    receipt_statement, receipt_identity, receipt_fingerprint = _authenticate_receipt(
        receipt
    )
    manifest_raw, manifest = _load_object(
        evidence / "manifest.json", label="evidence manifest"
    )
    if receipt_statement.get("pack_manifest_digest") != _digest(manifest_raw):
        raise AcceptanceAttestationError(
            "receipt does not bind the supplied evidence manifest"
        )
    evidence_references = manifest.get("evidence")
    if not isinstance(evidence_references, dict):
        raise AcceptanceAttestationError("evidence manifest references are invalid")
    baseline_raw, baseline_identity = _bound_object(
        evidence,
        evidence_references.get("baseline_provider_identity"),
        label="baseline artifact identity",
    )
    subject_raw, subject_identity = _bound_object(
        evidence,
        evidence_references.get("subject_provider_identity"),
        label="subject artifact identity",
    )
    schedule_raw, schedule = _bound_object(
        evidence,
        evidence_references.get("schedule"),
        label="evaluation schedule",
    )
    _report_raw, report = _bound_object(
        evidence,
        evidence_references.get("evaluation_report"),
        label="comparison report",
    )
    _request_raw, request = _bound_object(
        evidence,
        evidence_references.get("request"),
        label="evaluation request",
    )
    anchors = receipt_statement.get("anchors")
    if not isinstance(anchors, dict):
        raise AcceptanceAttestationError("receipt anchors are invalid")
    artifact_anchors = anchors.get("artifact_digests")
    if not isinstance(artifact_anchors, dict):
        raise AcceptanceAttestationError("receipt artifact anchors are invalid")
    baseline = _artifact(
        baseline_raw,
        baseline_identity,
        anchor=artifact_anchors.get("baseline"),
    )
    subject = _artifact(
        subject_raw,
        subject_identity,
        anchor=artifact_anchors.get("subject"),
    )
    schedule_digest = _digest(schedule_raw)
    if anchors.get("schedule_digest") != schedule_digest:
        raise AcceptanceAttestationError(
            "evaluation schedule does not match receipt anchor"
        )
    dataset_identity = schedule.get("dataset_identity")
    if not isinstance(dataset_identity, dict) or not dataset_identity:
        raise AcceptanceAttestationError(
            "evaluation schedule source identity is invalid"
        )
    verdict = receipt_statement.get("verdict")
    assert isinstance(verdict, dict)
    if verdict.get("integrity_ok") is not True:
        raise AcceptanceAttestationError(
            "only integrity-verified receipts can be transported"
        )
    if verdict.get("policy_verdict") not in {"pass", "fail"}:
        raise AcceptanceAttestationError(
            "receipt has no completed technical policy verdict"
        )
    if report.get("verdict") != verdict.get("policy_verdict"):
        raise AcceptanceAttestationError(
            "comparison report verdict disagrees with receipt"
        )
    if report.get("policy_digest") != anchors.get("policy_digest"):
        raise AcceptanceAttestationError(
            "comparison report policy disagrees with receipt"
        )
    key = _load_private_key(Path(signing_key_path))
    envelope_fingerprint = public_key_fingerprint(key.public_key())
    envelope_identity = _safe_identity(
        signer_identity, label="attestation signer identity"
    )
    relationship = (
        "same_signer"
        if envelope_fingerprint == receipt_fingerprint
        and envelope_identity == receipt_identity
        else "countersigned"
    )
    predicate: dict[str, Any] = {
        "format": ACCEPTANCE_PREDICATE_FORMAT,
        "subject": subject,
        "baseline": baseline,
        "contracts": {
            "invarlock_release": _contract_release(
                receipt_statement.get("format"),
                manifest.get("format"),
                report.get("format"),
            ),
            "evidence_pack": manifest.get("format"),
            "comparison_report": report.get("format"),
            "receipt": receipt_statement.get("format"),
        },
        "evaluation_source": {
            "schedule_format": schedule.get("format_version"),
            "schedule_digest": schedule_digest,
            "identity_digest": _digest(_canonical_json_bytes(dataset_identity)),
            "identity": dataset_identity,
        },
        "metric": _metric(request, report),
        "policy": {
            "identity": _safe_identity(
                policy_identity, label="evaluated policy identity"
            ),
            "digest": anchors.get("policy_digest"),
        },
        "technical_verdict": dict(verdict),
        "timestamps": {
            "evaluation_completed_at": _timestamp(
                evaluation_completed_at, label="evaluation completion timestamp"
            ),
            # v0.13 receipts did not carry an issuance timestamp.  Preserve that
            # absence instead of manufacturing historical metadata.
            "receipt_issued_at": None,
            "attestation_issued_at": _timestamp(
                issued_at, label="attestation issuance timestamp"
            ),
        },
        "signers": {
            "receipt": {
                "identity": receipt_identity,
                "fingerprint": receipt_fingerprint,
            },
            "envelope": {
                "identity": envelope_identity,
                "fingerprint": envelope_fingerprint,
            },
            "relationship": relationship,
        },
        "receipt": {
            "representation": "embedded",
            "media_type": RECEIPT_MEDIA_TYPE,
            "digest": _digest(receipt_raw),
            "raw_base64": base64.b64encode(receipt_raw).decode("ascii"),
            "content": receipt,
        },
    }
    try:
        jsonschema.Draft202012Validator(load_acceptance_predicate_schema()).validate(
            predicate
        )
    except jsonschema.ValidationError as exc:
        raise AcceptanceAttestationError(
            f"acceptance predicate is invalid: {exc.message}"
        ) from exc
    subject_digest = subject["artifact_digest"].removeprefix("sha256:")
    statement = {
        "_type": IN_TOTO_STATEMENT_TYPE,
        "subject": [
            {
                "name": subject["name"],
                "digest": {"sha256": subject_digest},
            }
        ],
        "predicateType": ACCEPTANCE_PREDICATE_TYPE,
        "predicate": predicate,
    }
    payload = _canonical_json_bytes(statement)
    signature = key.sign(_dsse_pae(DSSE_PAYLOAD_TYPE, payload))
    envelope = {
        "payloadType": DSSE_PAYLOAD_TYPE,
        "payload": base64.b64encode(payload).decode("ascii"),
        "signatures": [
            {
                "keyid": envelope_fingerprint,
                "sig": base64.b64encode(signature).decode("ascii"),
            }
        ],
    }
    output = Path(output_path)
    _write_no_clobber(output, _canonical_json_bytes(envelope))
    return AcceptanceAttestation(
        path=output.resolve(),
        statement=statement,
        signer_fingerprint=envelope_fingerprint,
    )


def _load_policy(
    policy: Mapping[str, Any] | Path,
) -> dict[str, Any]:
    if isinstance(policy, Path):
        _raw, value = _load_object(policy, label="recipient acceptance policy")
        return value
    if not isinstance(policy, Mapping):
        raise AcceptanceAttestationError(
            "recipient acceptance policy must be an object"
        )
    return dict(policy)


def _validate_policy_trust_registries(policy: dict[str, Any]) -> None:
    for registry_name in ("trusted_signers", "trusted_receipt_verifiers"):
        seen: set[tuple[str, str]] = set()
        for record in policy[registry_name]:
            trust_key = (record["identity"], record["fingerprint"])
            if trust_key in seen:
                raise AcceptanceAttestationError(
                    f"{registry_name} contains a duplicate identity/fingerprint pair"
                )
            seen.add(trust_key)


def _technical_verdict(predicate: object) -> str | None:
    if not isinstance(predicate, dict):
        return None
    verdict = predicate.get("technical_verdict")
    value = verdict.get("policy_verdict") if isinstance(verdict, dict) else None
    return value if value in {"pass", "fail"} else None


def _file_sha256(path: Path) -> str:
    if path.is_symlink():
        raise AcceptanceAttestationError("subject artifact must not be a symlink")
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise AcceptanceAttestationError(
                "file-bound subject artifact must be a regular file"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise AcceptanceAttestationError(
            "subject artifact could not be read safely"
        ) from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise AcceptanceAttestationError(
            "subject artifact changed while it was being hashed"
        )
    return "sha256:" + digest.hexdigest()


def _artifact_path_digest(path: Path, *, digest_kind: object) -> str:
    if digest_kind == "file_sha256":
        return _file_sha256(path)
    if digest_kind == "hf_snapshot_tree_sha256":
        return checkpoint_tree_sha256(path)
    if digest_kind == "tensorrt_llm_engine_tree_sha256":
        return "sha256:" + read_tensorrt_llm_engine_tree_sha256(path)
    raise AcceptanceAttestationError("subject artifact digest kind is unsupported")


def _authenticated_statement(
    envelope_path: Path,
    trusted_public_keys: Mapping[str, Path | bytes | ed25519.Ed25519PublicKey],
) -> tuple[dict[str, Any], str]:
    _raw, envelope = _load_object(Path(envelope_path), label="DSSE envelope")
    if set(envelope) != {"payloadType", "payload", "signatures"}:
        raise AcceptanceAttestationError("DSSE envelope fields are invalid")
    if envelope.get("payloadType") != DSSE_PAYLOAD_TYPE:
        raise AcceptanceAttestationError("DSSE payload type is invalid")
    signatures = envelope.get("signatures")
    if not isinstance(signatures, list) or len(signatures) != 1:
        raise AcceptanceAttestationError(
            "DSSE envelope must contain exactly one signature"
        )
    signature = signatures[0]
    if not isinstance(signature, dict) or set(signature) != {"keyid", "sig"}:
        raise AcceptanceAttestationError("DSSE signature fields are invalid")
    keyid = _normalized_digest(signature.get("keyid"), label="DSSE signer key ID")
    encoded_payload = envelope.get("payload")
    encoded_signature = signature.get("sig")
    if not isinstance(encoded_payload, str) or not isinstance(encoded_signature, str):
        raise AcceptanceAttestationError("DSSE base64 fields are invalid")
    try:
        payload = base64.b64decode(encoded_payload, validate=True)
        signature_bytes = base64.b64decode(encoded_signature, validate=True)
    except ValueError as exc:
        raise AcceptanceAttestationError("DSSE base64 fields are invalid") from exc
    parsed = parse_json_bytes(payload, label="in-toto Statement")
    if not isinstance(parsed, dict):
        raise AcceptanceAttestationError("in-toto Statement must be a JSON object")
    statement = parsed
    if payload != _canonical_json_bytes(statement):
        raise AcceptanceAttestationError("in-toto Statement must use canonical JSON")
    key_material = trusted_public_keys.get(keyid)
    if key_material is None:
        raise AcceptanceAttestationError(
            "DSSE signer key is unavailable from recipient trust anchors"
        )
    public_key = _load_public_key(key_material)
    if public_key_fingerprint(public_key) != keyid:
        raise AcceptanceAttestationError(
            "recipient public key does not match DSSE key ID"
        )
    try:
        public_key.verify(
            signature_bytes,
            _dsse_pae(DSSE_PAYLOAD_TYPE, payload),
        )
    except InvalidSignature as exc:
        raise AcceptanceAttestationError("DSSE signature verification failed") from exc
    return statement, keyid


def _receipt_consistency_errors(
    predicate: dict[str, Any],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    receipt_block = predicate["receipt"]
    raw_matches_content = False
    try:
        receipt_raw = base64.b64decode(
            receipt_block["raw_base64"],
            validate=True,
        )
        digest_matches = receipt_block["digest"] == _digest(receipt_raw)
        if not digest_matches:
            errors.append("embedded receipt digest over raw bytes is invalid")
        parsed_receipt = parse_json_bytes(
            receipt_raw,
            label="embedded receipt raw bytes",
        )
        if parsed_receipt != receipt_block["content"]:
            errors.append("embedded receipt raw bytes disagree with content")
        else:
            raw_matches_content = digest_matches
    except (StrictJsonError, TypeError, ValueError):
        errors.append("embedded receipt raw bytes are invalid")
    try:
        receipt_statement, receipt_identity, receipt_fingerprint = (
            _authenticate_receipt(receipt_block["content"])
        )
        authenticated = raw_matches_content
    except AcceptanceAttestationError as exc:
        errors.append(str(exc).replace("embedded ", "", 1))
        receipt_statement = {}
        receipt_identity = ""
        receipt_fingerprint = ""
        authenticated = False
    if predicate["signers"]["receipt"] != {
        "identity": receipt_identity,
        "fingerprint": receipt_fingerprint,
    }:
        errors.append("predicate receipt signer disagrees with embedded receipt")
    if predicate["contracts"]["receipt"] != receipt_statement.get("format"):
        errors.append("predicate receipt contract disagrees with embedded receipt")
    if predicate["technical_verdict"] != receipt_statement.get("verdict"):
        errors.append("technical verdict disagrees with embedded receipt")
    if (
        receipt_statement.get("format") in _SUPPORTED_V013_RECEIPTS
        and predicate["timestamps"]["receipt_issued_at"] is not None
    ):
        errors.append(
            "receipt issuance timestamp is not authenticated by the embedded "
            "v0.13 receipt"
        )
    receipt_anchors = receipt_statement.get("anchors")
    if not isinstance(receipt_anchors, dict):
        return authenticated, errors
    artifact_anchors = receipt_anchors.get("artifact_digests")
    if not isinstance(artifact_anchors, dict):
        errors.append("embedded receipt artifact anchors are invalid")
    else:
        for role in ("baseline", "subject"):
            transported = predicate[role]
            try:
                identity_raw = base64.b64decode(
                    transported["artifact_identity_payload"],
                    validate=True,
                )
                parsed_identity = parse_json_bytes(
                    identity_raw,
                    label=f"{role} transported artifact identity",
                )
                if parsed_identity != transported["artifact_identity"]:
                    raise AcceptanceAttestationError(
                        f"{role} artifact identity payload disagrees with its object"
                    )
                reconstructed = _artifact(
                    identity_raw,
                    transported["artifact_identity"],
                    anchor=artifact_anchors.get(role),
                )
            except (
                AcceptanceAttestationError,
                StrictJsonError,
                ValueError,
            ) as exc:
                errors.append(str(exc))
            else:
                if transported != reconstructed:
                    errors.append(f"{role} artifact disagrees with embedded identity")
    if predicate["evaluation_source"]["schedule_digest"] != receipt_anchors.get(
        "schedule_digest"
    ):
        errors.append("evaluation source digest disagrees with embedded receipt")
    if predicate["policy"]["digest"] != receipt_anchors.get("policy_digest"):
        errors.append("policy digest disagrees with embedded receipt")
    return authenticated, errors


def _subject_binding_errors(
    statement: dict[str, Any],
    predicate: dict[str, Any],
    *,
    expected_subject_digest: str | None,
    subject_artifact_path: Path | None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    subject = predicate["subject"]
    expected_in_toto = [
        {
            "name": subject["name"],
            "digest": {"sha256": subject["artifact_digest"].removeprefix("sha256:")},
        }
    ]
    if statement.get("subject") != expected_in_toto:
        errors.append("in-toto subject disagrees with acceptance predicate")
    if (expected_subject_digest is None) == (subject_artifact_path is None):
        raise AcceptanceAttestationError(
            "provide exactly one expected subject digest or subject artifact path"
        )
    if expected_subject_digest is not None:
        observed = _normalized_digest(
            expected_subject_digest, label="expected subject digest"
        )
    else:
        assert subject_artifact_path is not None
        observed = _artifact_path_digest(
            Path(subject_artifact_path),
            digest_kind=subject["digest_kind"],
        )
    bound = subject["artifact_digest"] == observed
    if not bound:
        errors.append("subject digest does not match recipient artifact")
    return bound, errors


def _recipient_policy_errors(
    statement: dict[str, Any],
    predicate: dict[str, Any],
    policy: dict[str, Any],
    *,
    keyid: str,
    now: datetime | None,
) -> list[str]:
    errors: list[str] = []
    if statement.get("_type") != IN_TOTO_STATEMENT_TYPE:
        errors.append("in-toto Statement type is invalid")
    if statement.get("predicateType") != ACCEPTANCE_PREDICATE_TYPE:
        errors.append("acceptance predicate type is invalid")
    outer_signer = predicate["signers"]["envelope"]
    if outer_signer["fingerprint"] != keyid:
        errors.append("predicate envelope signer disagrees with DSSE key ID")
    trusted_signers = [
        item
        for item in policy["trusted_signers"]
        if item["fingerprint"] == keyid and item["identity"] == outer_signer["identity"]
    ]
    if not trusted_signers:
        errors.append("envelope signer is not trusted by recipient policy")
    elif len(trusted_signers) != 1:
        errors.append("envelope signer has multiple matching recipient trust records")
    elif trusted_signers[0]["status"] == "revoked":
        errors.append("envelope signer is revoked by recipient policy")
    receipt_signer = predicate["signers"]["receipt"]
    trusted_receipt_verifiers = [
        item
        for item in policy["trusted_receipt_verifiers"]
        if item["fingerprint"] == receipt_signer["fingerprint"]
        and item["identity"] == receipt_signer["identity"]
    ]
    if not trusted_receipt_verifiers:
        errors.append("receipt verifier is not trusted by recipient policy")
    elif len(trusted_receipt_verifiers) != 1:
        errors.append("receipt verifier has multiple matching recipient trust records")
    elif trusted_receipt_verifiers[0]["status"] == "revoked":
        errors.append("receipt verifier is revoked by recipient policy")
    expected_trust_profile_digest = policy.get("expected_receipt_trust_profile_digest")
    receipt_content = predicate["receipt"]["content"]
    receipt_statement = receipt_content.get("statement")
    receipt_verifier = (
        receipt_statement.get("verifier")
        if isinstance(receipt_statement, dict)
        else None
    )
    receipt_trust_profile_digest = (
        receipt_verifier.get("trust_profile_digest")
        if isinstance(receipt_verifier, dict)
        else None
    )
    if (
        expected_trust_profile_digest is not None
        and receipt_trust_profile_digest != expected_trust_profile_digest
    ):
        errors.append("receipt trust-profile digest does not satisfy recipient policy")
    if policy["expected_predicate_type"] != statement.get("predicateType"):
        errors.append("predicate type is not allowed by recipient policy")
    if (
        predicate["contracts"]["invarlock_release"]
        not in policy["allowed_contract_versions"]
    ):
        errors.append("contract version is not allowed by recipient policy")
    technical = predicate["technical_verdict"]
    if (
        technical["policy_verdict"] != policy["required_technical_verdict"]
        or technical["ok"] is not True
        or technical["integrity_ok"] is not True
    ):
        errors.append("technical verdict does not satisfy recipient policy")
    relationship = predicate["signers"]["relationship"]
    actual_relationship = (
        "same_signer"
        if predicate["signers"]["receipt"] == predicate["signers"]["envelope"]
        else "countersigned"
    )
    if relationship != actual_relationship:
        errors.append("declared signer relationship is inconsistent")
    if relationship == "countersigned" and not policy["allow_countersigned_receipts"]:
        errors.append("countersigned receipts are not allowed by recipient policy")
    envelope_issued_at = _parse_timestamp(
        predicate["timestamps"]["attestation_issued_at"],
        label="attestation issuance timestamp",
    )
    current = now or datetime.now(tz=UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise AcceptanceAttestationError(
            "recipient current time must include a timezone"
        )
    current = current.astimezone(UTC)
    skew = policy["freshness"]["clock_skew_seconds"]
    envelope_age = (current - envelope_issued_at).total_seconds()
    if envelope_age < -skew:
        errors.append("attestation issuance timestamp is in the future")
    if envelope_age > policy["freshness"]["max_envelope_age_seconds"] + skew:
        errors.append("attestation envelope is stale under recipient policy")
    max_evidence_age = policy["freshness"]["max_evidence_age_seconds"]
    if max_evidence_age is not None:
        receipt_issued_at = predicate["timestamps"]["receipt_issued_at"]
        if receipt_issued_at is None:
            errors.append(
                "authoritative evidence timestamp is unavailable under recipient policy"
            )
        else:
            evidence_issued_at = _parse_timestamp(
                receipt_issued_at,
                label="authoritative evidence timestamp",
            )
            evidence_age = (current - evidence_issued_at).total_seconds()
            if evidence_age < -skew:
                errors.append("authoritative evidence timestamp is in the future")
            if evidence_age > max_evidence_age + skew:
                errors.append("technical evidence is stale under recipient policy")
    return errors


def verify_acceptance_attestation(
    envelope_path: Path,
    *,
    trusted_public_keys: Mapping[str, Path | bytes | ed25519.Ed25519PublicKey],
    recipient_policy: Mapping[str, Any] | Path,
    expected_subject_digest: str | None = None,
    subject_artifact_path: Path | None = None,
    now: datetime | None = None,
) -> AcceptanceDecision:
    """Authenticate an envelope and apply one independently supplied policy."""

    errors: list[str] = []
    envelope_authenticated = False
    receipt_authenticated = False
    subject_bound = False
    statement: dict[str, Any] | None = None
    predicate: dict[str, Any] | None = None
    try:
        statement, keyid = _authenticated_statement(
            Path(envelope_path), trusted_public_keys
        )
        envelope_authenticated = True
        policy = _load_policy(recipient_policy)
        jsonschema.Draft202012Validator(
            load_recipient_acceptance_policy_schema()
        ).validate(policy)
        _validate_policy_trust_registries(policy)
        if set(statement) != {"_type", "subject", "predicateType", "predicate"}:
            raise AcceptanceAttestationError("in-toto Statement fields are invalid")
        predicate_value = statement.get("predicate")
        if not isinstance(predicate_value, dict):
            raise AcceptanceAttestationError("acceptance predicate is invalid")
        predicate = predicate_value
        jsonschema.Draft202012Validator(load_acceptance_predicate_schema()).validate(
            predicate
        )
        receipt_authenticated, receipt_errors = _receipt_consistency_errors(predicate)
        errors.extend(receipt_errors)
        subject_bound, subject_errors = _subject_binding_errors(
            statement,
            predicate,
            expected_subject_digest=expected_subject_digest,
            subject_artifact_path=subject_artifact_path,
        )
        errors.extend(subject_errors)
        errors.extend(
            _recipient_policy_errors(
                statement,
                predicate,
                policy,
                keyid=keyid,
                now=now,
            )
        )
    except (
        AcceptanceAttestationError,
        StrictJsonError,
        jsonschema.ValidationError,
    ) as exc:
        message = (
            exc.message if isinstance(exc, jsonschema.ValidationError) else str(exc)
        )
        errors.append(message)
    verdict = _technical_verdict(predicate)
    accepted = (
        envelope_authenticated
        and receipt_authenticated
        and subject_bound
        and not errors
    )
    return AcceptanceDecision(
        envelope_authenticated=envelope_authenticated,
        receipt_authenticated=receipt_authenticated,
        subject_bound=subject_bound,
        historical_technical_verdict=verdict,
        accepted=accepted,
        statement=statement,
        errors=tuple(errors),
    )


__all__ = [
    "ACCEPTANCE_PREDICATE_FORMAT",
    "ACCEPTANCE_PREDICATE_TYPE",
    "DSSE_PAYLOAD_TYPE",
    "IN_TOTO_STATEMENT_TYPE",
    "RECIPIENT_POLICY_FORMAT",
    "AcceptanceAttestation",
    "AcceptanceAttestationError",
    "AcceptanceDecision",
    "verify_acceptance_attestation",
    "write_acceptance_attestation",
]
