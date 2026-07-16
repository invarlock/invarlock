"""Independent signatures for evidence-pack-v1 verification receipts."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.evidence_pack_integrity import (
    normalize_expected_fingerprint,
    public_key_fingerprint,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.evidence_pack_support import EvidencePackResult

SIGNED_RECEIPT_FORMAT = "invarlock/evidence-verification-receipt-v1"
SIGNED_RECEIPT_SIGNATURE_FORMAT = "invarlock/evidence-verification-receipt-signature-v1"
_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_DIGEST_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")
_MAX_RECEIPT_BYTES = 1024 * 1024


class EvidenceReceiptError(ValueError):
    """Raised when a signed verification receipt is unsafe or untrusted."""


@dataclass(frozen=True)
class ReceiptVerification:
    ok: bool
    signed: bool
    statement: dict[str, Any] | None
    verifier_fingerprint: str | None
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
        raise EvidenceReceiptError(f"receipt is not canonical JSON: {exc}") from exc


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _normalized_digest(value: str | None, *, label: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if _DIGEST_RE.fullmatch(normalized) is None:
        raise EvidenceReceiptError(f"{label} must be a sha256:... digest")
    return normalized


def _safe_identity(value: str, *, label: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _IDENTITY_RE.fullmatch(normalized) is None:
        raise EvidenceReceiptError(f"{label} is invalid")
    return normalized


def _load_private_key(path: Path) -> ed25519.Ed25519PrivateKey:
    try:
        payload = read_regular_file_bytes(
            path,
            label="verification receipt signing key",
            max_bytes=64 * 1024,
        )
        key = serialization.load_pem_private_key(payload, password=None)
    except (StrictJsonError, TypeError, ValueError) as exc:
        raise EvidenceReceiptError(
            f"could not load receipt signing key: {exc}"
        ) from exc
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise EvidenceReceiptError("verification receipt signing key must be Ed25519")
    return key


def _outside_pack(pack_dir: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(pack_dir.resolve())
    except ValueError:
        return True
    return False


def _write_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise EvidenceReceiptError(f"receipt destination already exists: {path}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise EvidenceReceiptError(f"could not write signed receipt: {exc}") from exc


def _statement(
    result: EvidencePackResult,
    *,
    pack_manifest_digest: str,
    policy_path: Path,
    expected_artifact_digests: dict[str, str],
    expected_schedule_digest: str,
    expected_runtime_digests: dict[str, str],
    expected_pack_signer_fingerprint: str,
    verifier_identity: str,
    verifier_fingerprint: str,
) -> dict[str, object]:
    policy = read_regular_file_bytes(
        policy_path, label="independent policy anchor", max_bytes=4 * 1024 * 1024
    )
    artifacts = {
        side: _normalized_digest(value, label=f"{side} artifact anchor")
        for side, value in sorted(expected_artifact_digests.items())
    }
    if set(artifacts) != {"baseline", "subject"}:
        raise EvidenceReceiptError(
            "artifact anchors must contain exactly baseline and subject"
        )
    schedule = _normalized_digest(expected_schedule_digest, label="schedule anchor")
    runtimes = {
        side: _normalized_digest(value, label=f"{side} runtime anchor")
        for side, value in sorted(expected_runtime_digests.items())
    }
    if set(runtimes) != {"baseline", "subject"}:
        raise EvidenceReceiptError(
            "runtime anchors must contain exactly baseline and subject"
        )
    pack_signer = normalize_expected_fingerprint(expected_pack_signer_fingerprint)
    if pack_signer is None:
        raise EvidenceReceiptError(
            "pack signer anchor must be a sha256:... fingerprint"
        )
    result_anchors = result.payload.get("anchors")
    if isinstance(result_anchors, dict):
        recorded_policy = result_anchors.get("policy_digest")
        recorded_artifacts = result_anchors.get("artifact_digests")
        recorded_schedule = result_anchors.get("schedule_digest")
        recorded_runtimes = result_anchors.get("runtime_digests")
        recorded_signer = result_anchors.get("signer_fingerprint")
        if recorded_policy is not None and recorded_policy != _digest(policy):
            raise EvidenceReceiptError(
                "verification result policy anchor does not match caller policy"
            )
        if recorded_artifacts is not None and recorded_artifacts != artifacts:
            raise EvidenceReceiptError(
                "verification result artifact anchors do not match caller artifacts"
            )
        if recorded_schedule is not None and recorded_schedule != schedule:
            raise EvidenceReceiptError(
                "verification result schedule anchor does not match caller schedule"
            )
        if recorded_signer is not None and recorded_signer != pack_signer:
            raise EvidenceReceiptError(
                "verification result signer anchor does not match caller signer"
            )
        if recorded_runtimes is not None and recorded_runtimes != runtimes:
            raise EvidenceReceiptError(
                "verification result runtime anchors do not match caller runtimes"
            )
    policy_verdict = result.payload.get("policy_verdict")
    if policy_verdict not in {"pass", "fail", None}:
        raise EvidenceReceiptError("verification result policy verdict is invalid")
    return {
        "format": SIGNED_RECEIPT_FORMAT,
        "pack_manifest_digest": pack_manifest_digest,
        "anchors": {
            "policy_digest": _digest(policy),
            "artifact_digests": artifacts,
            "schedule_digest": schedule,
            "runtime_digests": runtimes,
            "pack_signer_fingerprint": pack_signer,
        },
        "verifier": {
            "identity": _safe_identity(verifier_identity, label="verifier identity"),
            "signing_key_fingerprint": verifier_fingerprint,
        },
        "verdict": {
            "ok": bool(result.payload.get("ok")),
            "integrity_ok": bool(result.payload.get("integrity_ok")),
            "policy_verdict": policy_verdict,
            "verification_status": int(result.status),
        },
    }


def write_signed_verification_receipt(
    pack_dir: Path,
    result: EvidencePackResult,
    receipt_path: Path,
    *,
    policy_path: Path,
    expected_artifact_digests: dict[str, str],
    expected_schedule_digest: str,
    expected_runtime_digests: dict[str, str],
    expected_pack_signer_fingerprint: str,
    verifier_identity: str,
    verifier_signing_key_path: Path,
) -> str:
    """Sign explicit caller anchors and a completed verification verdict."""

    pack_dir = Path(pack_dir)
    receipt_path = Path(receipt_path)
    if not _outside_pack(pack_dir, receipt_path):
        raise EvidenceReceiptError(
            "signed receipt must remain outside the evidence pack"
        )
    manifest_digest = _normalized_digest(
        result.manifest_digest,
        label="verification result manifest digest",
    )
    private_key = _load_private_key(verifier_signing_key_path)
    public_key = private_key.public_key()
    fingerprint = public_key_fingerprint(public_key)
    statement = _statement(
        result,
        pack_manifest_digest=manifest_digest,
        policy_path=Path(policy_path),
        expected_artifact_digests=expected_artifact_digests,
        expected_schedule_digest=expected_schedule_digest,
        expected_runtime_digests=expected_runtime_digests,
        expected_pack_signer_fingerprint=expected_pack_signer_fingerprint,
        verifier_identity=verifier_identity,
        verifier_fingerprint=fingerprint,
    )
    statement_bytes = _canonical_json_bytes(statement)
    receipt = {
        "statement": statement,
        "signature": {
            "format": SIGNED_RECEIPT_SIGNATURE_FORMAT,
            "algorithm": "ed25519",
            "public_key": {
                "encoding": "pem",
                "value": public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                ).decode("ascii"),
            },
            "value": base64.b64encode(private_key.sign(statement_bytes)).decode(
                "ascii"
            ),
        },
    }
    _write_no_clobber(receipt_path, _canonical_json_bytes(receipt))
    return fingerprint


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        raw = read_regular_file_bytes(
            path, label="verification receipt", max_bytes=_MAX_RECEIPT_BYTES
        )
        payload = parse_json_bytes(raw, label="verification receipt")
    except StrictJsonError as exc:
        raise EvidenceReceiptError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise EvidenceReceiptError("verification receipt must be a JSON object")
    return payload


def verify_signed_verification_receipt(
    receipt_path: Path,
    pack_dir: Path,
    *,
    policy_path: Path,
    expected_artifact_digests: dict[str, str],
    expected_schedule_digest: str,
    expected_runtime_digests: dict[str, str],
    expected_pack_signer_fingerprint: str,
    expected_verifier_identity: str,
    expected_verifier_fingerprint: str,
    require_signed: bool = True,
) -> ReceiptVerification:
    """Verify a receipt only against independently supplied trust anchors."""

    errors: list[str] = []
    receipt_path = Path(receipt_path)
    pack_dir = Path(pack_dir)
    if not _outside_pack(pack_dir, receipt_path):
        errors.append("verification receipt is inside the evidence pack")
    try:
        receipt = _load_receipt(receipt_path)
    except EvidenceReceiptError as exc:
        return ReceiptVerification(False, False, None, None, (str(exc),))
    statement = receipt.get("statement")
    signature = receipt.get("signature")
    signed = isinstance(statement, dict) and isinstance(signature, dict)
    if not signed:
        message = "signed verification receipt is required"
        return ReceiptVerification(
            False,
            False,
            None,
            None,
            tuple([*errors, message] if require_signed else errors),
        )
    assert isinstance(statement, dict) and isinstance(signature, dict)
    expected_statement_fields = {
        "format",
        "pack_manifest_digest",
        "anchors",
        "verifier",
        "verdict",
    }
    if set(statement) != expected_statement_fields:
        errors.append("signed receipt statement fields are invalid")
    if statement.get("format") != SIGNED_RECEIPT_FORMAT:
        errors.append("signed receipt format is invalid")
    manifest_claim = statement.get("pack_manifest_digest")
    if (
        not isinstance(manifest_claim, str)
        or _DIGEST_RE.fullmatch(manifest_claim) is None
    ):
        errors.append("signed receipt pack manifest digest is invalid")
    if set(signature) != {"format", "algorithm", "public_key", "value"}:
        errors.append("signed receipt signature fields are invalid")
    if signature.get("format") != SIGNED_RECEIPT_SIGNATURE_FORMAT:
        errors.append("signed receipt signature format is invalid")
    if signature.get("algorithm") != "ed25519":
        errors.append("signed receipt algorithm must be ed25519")

    external_verifier = normalize_expected_fingerprint(expected_verifier_fingerprint)
    if external_verifier is None:
        errors.append("expected verifier fingerprint is invalid")
    verifier = statement.get("verifier")
    if not isinstance(verifier, dict) or set(verifier) != {
        "identity",
        "signing_key_fingerprint",
    }:
        errors.append("signed receipt verifier fields are invalid")
    recorded_fingerprint = (
        verifier.get("signing_key_fingerprint") if isinstance(verifier, dict) else None
    )
    recorded_identity = verifier.get("identity") if isinstance(verifier, dict) else None
    try:
        expected_identity = _safe_identity(
            expected_verifier_identity, label="expected verifier identity"
        )
    except EvidenceReceiptError as exc:
        errors.append(str(exc))
        expected_identity = None
    if recorded_identity != expected_identity:
        errors.append("receipt verifier identity does not match caller expectation")

    verdict = statement.get("verdict")
    if not isinstance(verdict, dict) or set(verdict) != {
        "ok",
        "integrity_ok",
        "policy_verdict",
        "verification_status",
    }:
        errors.append("signed receipt verdict fields are invalid")
    else:
        if not isinstance(verdict.get("ok"), bool) or not isinstance(
            verdict.get("integrity_ok"), bool
        ):
            errors.append("signed receipt verdict booleans are invalid")
        if verdict.get("policy_verdict") not in {"pass", "fail", None}:
            errors.append("signed receipt policy verdict is invalid")
        status = verdict.get("verification_status")
        if isinstance(status, bool) or not isinstance(status, int) or status < 0:
            errors.append("signed receipt verification status is invalid")
        if verdict.get("ok") is True and (
            verdict.get("integrity_ok") is not True
            or verdict.get("policy_verdict") == "fail"
            or status != 0
        ):
            errors.append("signed receipt successful verdict is inconsistent")

    public_key_block = signature.get("public_key")
    public_key_value = (
        public_key_block.get("value")
        if isinstance(public_key_block, dict)
        and public_key_block.get("encoding") == "pem"
        else None
    )
    public_key: ed25519.Ed25519PublicKey | None = None
    if not isinstance(public_key_value, str):
        errors.append("signed receipt public key is invalid")
    else:
        try:
            loaded = serialization.load_pem_public_key(public_key_value.encode("ascii"))
            if not isinstance(loaded, ed25519.Ed25519PublicKey):
                raise TypeError("public key is not Ed25519")
            public_key = loaded
        except (TypeError, ValueError) as exc:
            errors.append(f"signed receipt public key is invalid: {exc}")
    derived_fingerprint = (
        public_key_fingerprint(public_key) if public_key is not None else None
    )
    if recorded_fingerprint != derived_fingerprint:
        errors.append("receipt verifier fingerprint does not match its public key")
    if derived_fingerprint != external_verifier:
        errors.append("receipt verifier key does not match caller expectation")
    if public_key is not None:
        encoded_signature = signature.get("value")
        try:
            if not isinstance(encoded_signature, str):
                raise ValueError("signature value is not text")
            signature_bytes = base64.b64decode(encoded_signature, validate=True)
            public_key.verify(signature_bytes, _canonical_json_bytes(statement))
        except (InvalidSignature, TypeError, ValueError):
            errors.append("signed receipt signature verification failed")

    try:
        manifest_digest = _digest(
            read_regular_file_bytes(
                pack_dir / "manifest.json",
                label="pack manifest",
                max_bytes=256 * 1024,
            )
        )
        policy_digest = _digest(
            read_regular_file_bytes(
                policy_path,
                label="independent policy anchor",
                max_bytes=4 * 1024 * 1024,
            )
        )
        artifact_digests = {
            side: _normalized_digest(value, label=f"{side} artifact anchor")
            for side, value in sorted(expected_artifact_digests.items())
        }
        if set(artifact_digests) != {"baseline", "subject"}:
            raise EvidenceReceiptError(
                "artifact anchors must contain exactly baseline and subject"
            )
        schedule_digest = _normalized_digest(
            expected_schedule_digest, label="schedule anchor"
        )
        runtime_digests = {
            side: _normalized_digest(value, label=f"{side} runtime anchor")
            for side, value in sorted(expected_runtime_digests.items())
        }
        if set(runtime_digests) != {"baseline", "subject"}:
            raise EvidenceReceiptError(
                "runtime anchors must contain exactly baseline and subject"
            )
        pack_signer = normalize_expected_fingerprint(expected_pack_signer_fingerprint)
        if pack_signer is None:
            raise EvidenceReceiptError("pack signer anchor is invalid")
        expected_anchors = {
            "policy_digest": policy_digest,
            "artifact_digests": artifact_digests,
            "schedule_digest": schedule_digest,
            "runtime_digests": runtime_digests,
            "pack_signer_fingerprint": pack_signer,
        }
        if statement.get("pack_manifest_digest") != manifest_digest:
            errors.append("receipt does not bind the supplied pack manifest")
        if statement.get("anchors") != expected_anchors:
            errors.append("receipt anchors do not match caller-supplied anchors")
    except (EvidenceReceiptError, StrictJsonError) as exc:
        errors.append(str(exc))

    return ReceiptVerification(
        ok=not errors,
        signed=True,
        statement=statement,
        verifier_fingerprint=derived_fingerprint,
        errors=tuple(errors),
    )


__all__ = [
    "EvidenceReceiptError",
    "ReceiptVerification",
    "SIGNED_RECEIPT_FORMAT",
    "verify_signed_verification_receipt",
    "write_signed_verification_receipt",
]
