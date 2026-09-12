"""Independent offline acceptance and signed receipts for bounded judge evidence.

The externally supplied recipient policy is the authority. Embedded evidence
signer keys are only evidence until their fingerprint and identity match that
policy. Local verification returns an unsigned result. A separate Ed25519
receipt can authenticate that complete result for transport without expanding
the bounded fixed-benchmark claim.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from jsonschema import Draft202012Validator, ValidationError

from invarlock.captured_contracts import atomic_write, secure_directory
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_json import read_regular_file_bytes
from invarlock.judge_evidence_types import JudgeRecipientPolicy
from invarlock.judge_measurements.contracts import canonical_payload
from invarlock.judge_measurements.evidence import (
    DECISION_SCOPE,
    JudgeEvidenceError,
    load_judge_evidence_envelope,
    object_sha256,
    read_object,
    replay_judge_evidence,
    signed_envelope_bytes,
)
from invarlock.judge_measurements.statistics import Decision
from invarlock.public_contracts import (
    load_judge_measurement_recipient_policy_schema,
    load_judge_measurement_verification_receipt_schema,
    load_judge_verification_result_schema,
)

RECIPIENT_POLICY_FORMAT: Final = "invarlock/judge-measurement-recipient-policy-v1"
RESULT_FORMAT: Final = "invarlock/judge-verification-result-v1"
RECEIPT_FORMAT: Final = "invarlock/judge-measurement-verification-receipt-v1"
RECEIPT_SIGNATURE_FORMAT: Final = (
    "invarlock/judge-measurement-verification-receipt-signature-v1"
)
_RECEIPT_DOMAIN: Final = b"invarlock:judge-measurement-verification-receipt:v1\x00"
_IDENTITY_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_FINGERPRINT_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OBJECT_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_MAX_CONTROL_BYTES: Final = 65536
_MAX_RECEIPT_BYTES: Final = 1024 * 1024


@dataclass(frozen=True)
class ReceiptBindings:
    baseline_run_sha256: str
    subject_run_sha256: str
    case_set_sha256: str
    plan_sha256: str
    measurements_sha256: str
    analysis_policy_sha256: str
    analysis_result_sha256: str


@dataclass(frozen=True)
class JudgeVerificationResult:
    """Unsigned result of one local recipient-policy verification and replay."""

    verified: bool = False
    authenticated: bool = False
    replayed: bool = False
    accepted: bool = False
    decision: Decision | None = None
    intended_subject: str | None = None
    signer_identity: str | None = None
    signer_public_key_sha256: str | None = None
    envelope_sha256: str | None = None
    recipient_policy_sha256: str | None = None
    bindings: ReceiptBindings | None = None
    errors: tuple[str, ...] = ()
    format: str = RESULT_FORMAT
    decision_scope: str = DECISION_SCOPE

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["errors"] = list(self.errors)
        return value


@dataclass(frozen=True)
class JudgeReceiptVerification:
    """Authentication result for one externally supplied signed judge receipt."""

    ok: bool
    signed: bool
    statement: dict[str, Any] | None
    result: JudgeVerificationResult | None
    verifier_fingerprint: str | None
    errors: tuple[str, ...]


def _require_external(path: Path, evidence: Path, *, label: str) -> None:
    absolute = Path(path).absolute()
    root = Path(evidence).absolute()
    if absolute.is_relative_to(root) or absolute.resolve().is_relative_to(
        root.resolve()
    ):
        raise JudgeEvidenceError(
            f"{label} must be supplied outside submitted judge evidence"
        )


def _safe_identity(value: str, *, label: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _IDENTITY_RE.fullmatch(normalized) is None:
        raise JudgeEvidenceError(f"{label} is invalid")
    return normalized


def _safe_fingerprint(value: str, *, label: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if _FINGERPRINT_RE.fullmatch(normalized) is None:
        raise JudgeEvidenceError(f"{label} must be a sha256:... fingerprint")
    return normalized


def _safe_object_digest(value: str, *, label: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if _OBJECT_DIGEST_RE.fullmatch(normalized) is None:
        raise JudgeEvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _result_from_dict(value: Mapping[str, Any]) -> JudgeVerificationResult:
    bindings = value.get("bindings")
    return JudgeVerificationResult(
        verified=bool(value["verified"]),
        authenticated=bool(value["authenticated"]),
        replayed=bool(value["replayed"]),
        accepted=bool(value["accepted"]),
        decision=cast(Decision | None, value["decision"]),
        intended_subject=cast(str | None, value["intended_subject"]),
        signer_identity=cast(str | None, value["signer_identity"]),
        signer_public_key_sha256=cast(str | None, value["signer_public_key_sha256"]),
        envelope_sha256=cast(str | None, value["envelope_sha256"]),
        recipient_policy_sha256=cast(str | None, value["recipient_policy_sha256"]),
        bindings=ReceiptBindings(**bindings) if isinstance(bindings, dict) else None,
        errors=tuple(cast(list[str], value["errors"])),
        format=cast(str, value["format"]),
        decision_scope=cast(str, value["decision_scope"]),
    )


def _validate_result(result: JudgeVerificationResult) -> dict[str, Any]:
    value = result.to_dict()
    try:
        Draft202012Validator(load_judge_verification_result_schema()).validate(value)
    except ValidationError as exc:
        raise JudgeEvidenceError(
            f"judge verification result is invalid: {exc.message[:240]}"
        ) from exc
    return value


def load_judge_recipient_policy(
    path: Path, *, evidence_path: Path
) -> JudgeRecipientPolicy:
    """Load the caller-selected external policy; never search evidence for trust."""
    path = Path(path).absolute()
    _require_external(path, Path(evidence_path), label="recipient policy")
    with secure_directory(path.parent):
        value = read_object(path, maximum=_MAX_CONTROL_BYTES)
    error = next(
        Draft202012Validator(
            load_judge_measurement_recipient_policy_schema()
        ).iter_errors(value),
        None,
    )
    if error is not None:
        raise JudgeEvidenceError(
            f"judge recipient policy is invalid: {error.message[:240]}"
        )
    return cast(JudgeRecipientPolicy, value)


def _additional_key(
    value: Path | bytes | Ed25519PublicKey, *, evidence: Path
) -> Ed25519PublicKey:
    if isinstance(value, Ed25519PublicKey):
        return value
    if isinstance(value, Path):
        _require_external(value, evidence, label="trusted signer key")
        path = value.absolute()
        with secure_directory(path.parent):
            value = read_regular_file_bytes(
                path, label="trusted signer key", max_bytes=_MAX_CONTROL_BYTES
            )
    if len(value) == 32:
        return Ed25519PublicKey.from_public_bytes(value)
    try:
        key = serialization.load_pem_public_key(value)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise JudgeEvidenceError(f"could not load trusted signer key: {exc}") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise JudgeEvidenceError("trusted signer key must be Ed25519")
    return key


def verify_judge_evidence(
    path: Path,
    *,
    recipient_policy_path: Path,
    trusted_public_keys: Mapping[str, Path | bytes | Ed25519PublicKey] | None = None,
) -> JudgeVerificationResult:
    """Authenticate, replay, and decide from independently supplied recipient pins."""
    root = Path(path).absolute()
    result = JudgeVerificationResult()
    try:
        policy = load_judge_recipient_policy(recipient_policy_path, evidence_path=root)
        result = replace(result, recipient_policy_sha256=object_sha256(policy))
        envelope = load_judge_evidence_envelope(root)
        result = replace(
            result,
            envelope_sha256=object_sha256(envelope),
            intended_subject=envelope["intended_subject"],
            bindings=ReceiptBindings(**envelope["bindings"]),
        )
        signer = envelope["signer"]
        if signer is None or envelope["signature"] is None:
            raise JudgeEvidenceError(
                "unsigned judge evidence cannot be independently accepted"
            )
        result = replace(
            result,
            signer_identity=signer["identity"],
            signer_public_key_sha256=signer["public_key_sha256"],
        )
        trusted = policy["trusted_signer"]
        if (
            signer["identity"] != trusted["identity"]
            or signer["public_key_sha256"] != trusted["public_key_sha256"]
        ):
            raise JudgeEvidenceError(
                "judge evidence signer is not authorized by recipient policy"
            )
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(signer["public_key"], validate=True)
        )
        fingerprint = public_key_fingerprint(public_key)
        if fingerprint != trusted["public_key_sha256"]:
            raise JudgeEvidenceError(
                "judge evidence signer key does not match recipient fingerprint"
            )
        if trusted_public_keys is not None:
            key_material = trusted_public_keys.get(trusted["identity"])
            if (
                key_material is None
                or public_key_fingerprint(_additional_key(key_material, evidence=root))
                != fingerprint
            ):
                raise JudgeEvidenceError(
                    "judge signer differs from the additional recipient key anchor"
                )
        try:
            public_key.verify(
                base64.b64decode(envelope["signature"], validate=True),
                signed_envelope_bytes(envelope),
            )
        except InvalidSignature as exc:
            raise JudgeEvidenceError("judge evidence signature is invalid") from exc
        result = replace(result, authenticated=True)
        if envelope["decision_scope"] != policy["decision_scope"]:
            raise JudgeEvidenceError(
                "judge evidence scope differs from recipient policy"
            )
        if envelope["intended_subject"] != policy["intended_subject"]:
            raise JudgeEvidenceError(
                "judge evidence intended subject differs from recipient policy"
            )
        for name, expected in policy["bindings"].items():
            if cast(Mapping[str, str], envelope["bindings"])[name] != expected:
                raise JudgeEvidenceError(
                    f"judge evidence {name} differs from recipient policy"
                )
        publication = replay_judge_evidence(
            root, expected_envelope_sha256=result.envelope_sha256
        )
        analysis = publication.analysis_result
        result = replace(result, replayed=True, decision=analysis.decision)
        if analysis.policy.metric_name != policy["required_metric_name"]:
            raise JudgeEvidenceError(
                "judge metric differs from the recipient's required metric"
            )
        result = replace(result, verified=True)
        if not analysis.policy.required:
            raise JudgeEvidenceError(
                "advisory judge evidence cannot satisfy a required recipient decision"
            )
        if analysis.decision != "pass":
            raise JudgeEvidenceError(f"required judge decision is {analysis.decision}")
        result = replace(result, accepted=True)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = replace(
            result,
            accepted=False,
            errors=(str(exc)[:1000] or "judge evidence verification failed",),
        )
    _validate_result(result)
    return result


def verify_judge_evidence_with_policy(
    path: Path, recipient_policy_path: Path
) -> JudgeVerificationResult:
    return verify_judge_evidence(path, recipient_policy_path=recipient_policy_path)


def _load_private_key(
    path: Path, *, evidence_path: Path, key_bytes: bytes | None
) -> Ed25519PrivateKey:
    _require_external(path, evidence_path, label="verifier signing key")
    if key_bytes is None:
        absolute = Path(path).absolute()
        with secure_directory(absolute.parent):
            payload = read_regular_file_bytes(
                absolute,
                label="verifier signing key",
                max_bytes=_MAX_CONTROL_BYTES,
            )
    else:
        if not isinstance(key_bytes, bytes):
            raise JudgeEvidenceError("verifier signing key bytes must be exact bytes")
        if len(key_bytes) > _MAX_CONTROL_BYTES:
            raise JudgeEvidenceError(
                "verifier signing key exceeds the 65536-byte size limit"
            )
        payload = key_bytes
    try:
        key = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise JudgeEvidenceError(f"could not load verifier signing key: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise JudgeEvidenceError("verifier signing key must be Ed25519")
    return key


def _signed_receipt_bytes(statement: Mapping[str, Any]) -> bytes:
    """Apply a judge-specific domain before the canonical signed statement."""
    return _RECEIPT_DOMAIN + canonical_payload(statement)


def write_signed_judge_verification_receipt(
    evidence_path: Path,
    result: JudgeVerificationResult,
    receipt_path: Path,
    *,
    recipient_policy_path: Path,
    verifier_identity: str,
    verifier_signing_key_path: Path,
    verifier_signing_key_bytes: bytes | None = None,
) -> str:
    """Write a scoped Ed25519 receipt over a complete validated local result."""
    root = Path(evidence_path).absolute()
    destination = Path(receipt_path).absolute()
    _require_external(destination, root, label="signed judge receipt")
    policy = load_judge_recipient_policy(recipient_policy_path, evidence_path=root)
    policy_digest = object_sha256(policy)
    result_value = _validate_result(result)
    if result.recipient_policy_sha256 != policy_digest:
        raise JudgeEvidenceError(
            "judge verification result does not bind the supplied recipient policy"
        )
    key = _load_private_key(
        Path(verifier_signing_key_path),
        evidence_path=root,
        key_bytes=verifier_signing_key_bytes,
    )
    public = key.public_key()
    fingerprint = public_key_fingerprint(public)
    statement = {
        "format": RECEIPT_FORMAT,
        "kind": "judge",
        "decision_scope": DECISION_SCOPE,
        "recipient_policy_sha256": policy_digest,
        "verifier": {
            "identity": _safe_identity(verifier_identity, label="verifier identity"),
            "signing_key_fingerprint": fingerprint,
        },
        "result": result_value,
    }
    receipt = {
        "statement": statement,
        "signature": {
            "format": RECEIPT_SIGNATURE_FORMAT,
            "algorithm": "ed25519",
            "public_key": {
                "encoding": "pem",
                "value": public.public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                ).decode("ascii"),
            },
            "value": base64.b64encode(
                key.sign(_signed_receipt_bytes(statement))
            ).decode("ascii"),
        },
    }
    Draft202012Validator(load_judge_measurement_verification_receipt_schema()).validate(
        receipt
    )
    raw = canonical_payload(receipt) + b"\n"
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise JudgeEvidenceError("signed judge receipt exceeds the 1048576-byte limit")
    atomic_write(destination, raw)
    return fingerprint


def verify_signed_judge_verification_receipt(
    receipt_path: Path,
    *,
    expected_verifier_identity: str,
    expected_verifier_fingerprint: str,
    expected_recipient_policy_sha256: str,
) -> JudgeReceiptVerification:
    """Authenticate a signed judge result against caller-owned verifier anchors."""
    errors: list[str] = []
    statement: dict[str, Any] | None = None
    result: JudgeVerificationResult | None = None
    derived: str | None = None
    signed = False
    try:
        receipt = read_object(Path(receipt_path), maximum=_MAX_RECEIPT_BYTES)
        signed = isinstance(receipt.get("statement"), dict) and isinstance(
            receipt.get("signature"), dict
        )
        Draft202012Validator(
            load_judge_measurement_verification_receipt_schema()
        ).validate(receipt)
        statement = cast(dict[str, Any], receipt["statement"])
        signature = cast(dict[str, Any], receipt["signature"])
        expected_identity = _safe_identity(
            expected_verifier_identity, label="expected verifier identity"
        )
        expected_fingerprint = _safe_fingerprint(
            expected_verifier_fingerprint,
            label="expected verifier fingerprint",
        )
        expected_policy_digest = _safe_object_digest(
            expected_recipient_policy_sha256,
            label="expected recipient policy digest",
        )
        if statement["recipient_policy_sha256"] != expected_policy_digest:
            raise JudgeEvidenceError(
                "judge receipt recipient policy does not match caller expectation"
            )
        verifier = statement["verifier"]
        if verifier["identity"] != expected_identity:
            raise JudgeEvidenceError(
                "judge receipt verifier identity does not match caller expectation"
            )
        public = serialization.load_pem_public_key(
            signature["public_key"]["value"].encode("ascii")
        )
        if not isinstance(public, Ed25519PublicKey):
            raise JudgeEvidenceError("judge receipt verifier key must be Ed25519")
        derived = public_key_fingerprint(public)
        if (
            derived != expected_fingerprint
            or verifier["signing_key_fingerprint"] != derived
        ):
            raise JudgeEvidenceError("judge receipt verifier key is unauthorized")
        public.verify(
            base64.b64decode(signature["value"], validate=True),
            _signed_receipt_bytes(statement),
        )
        result_value = cast(dict[str, Any], statement["result"])
        if (
            statement["recipient_policy_sha256"]
            != result_value["recipient_policy_sha256"]
        ):
            raise JudgeEvidenceError(
                "judge receipt policy digest differs from its verification result"
            )
        result = _result_from_dict(result_value)
        _validate_result(result)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        ValidationError,
        InvalidSignature,
        UnsupportedAlgorithm,
    ) as exc:
        errors.append(str(exc)[:1000] or "judge receipt signature is invalid")
        statement = None
        result = None
        derived = None
    return JudgeReceiptVerification(
        ok=not errors,
        signed=signed,
        statement=statement,
        result=result,
        verifier_fingerprint=derived,
        errors=tuple(errors),
    )


def verify_stored_judge_result(
    stored_result_path: Path,
    *,
    evidence_path: Path,
    recipient_policy_path: Path,
) -> JudgeVerificationResult:
    """Recompute a stored local result; an asserted accepted flag is never authority."""
    root = Path(evidence_path).absolute()
    try:
        _require_external(stored_result_path, root, label="stored judge result")
    except (OSError, ValueError) as exc:
        rejected = JudgeVerificationResult(errors=(str(exc)[:1000],))
        _validate_result(rejected)
        return rejected
    actual = verify_judge_evidence_with_policy(root, recipient_policy_path)
    try:
        claimed = read_object(Path(stored_result_path), maximum=_MAX_CONTROL_BYTES)
        Draft202012Validator(load_judge_verification_result_schema()).validate(claimed)
        if object_sha256(claimed) != object_sha256(actual.to_dict()):
            raise JudgeEvidenceError(
                "stored judge result differs from independent recipient verification"
            )
    except (OSError, ValueError, TypeError, ValidationError) as exc:
        return replace(
            actual, accepted=False, verified=False, errors=(str(exc)[:1000],)
        )
    return actual


def replay_signed_judge_verification_receipt(
    receipt_path: Path,
    *,
    evidence_path: Path,
    recipient_policy_path: Path,
    expected_verifier_identity: str,
    expected_verifier_fingerprint: str,
) -> JudgeVerificationResult:
    """Authenticate a stored receipt and require exact fresh evidence replay."""
    root = Path(evidence_path).absolute()
    try:
        _require_external(receipt_path, root, label="signed judge receipt")
    except (OSError, ValueError) as exc:
        rejected = JudgeVerificationResult(errors=(str(exc)[:1000],))
        _validate_result(rejected)
        return rejected
    actual = verify_judge_evidence_with_policy(root, recipient_policy_path)
    try:
        policy = load_judge_recipient_policy(recipient_policy_path, evidence_path=root)
        verification = verify_signed_judge_verification_receipt(
            receipt_path,
            expected_verifier_identity=expected_verifier_identity,
            expected_verifier_fingerprint=expected_verifier_fingerprint,
            expected_recipient_policy_sha256=object_sha256(policy),
        )
        if not verification.ok or verification.result is None:
            raise JudgeEvidenceError(
                "; ".join(verification.errors) or "judge receipt is not authentic"
            )
        if object_sha256(verification.result.to_dict()) != object_sha256(
            actual.to_dict()
        ):
            raise JudgeEvidenceError(
                "signed judge receipt differs from fresh recipient verification"
            )
    except (OSError, ValueError, TypeError, ValidationError) as exc:
        return replace(
            actual, accepted=False, verified=False, errors=(str(exc)[:1000],)
        )
    return actual


__all__ = [
    "JudgeReceiptVerification",
    "JudgeVerificationResult",
    "RECEIPT_FORMAT",
    "RECEIPT_SIGNATURE_FORMAT",
    "RECIPIENT_POLICY_FORMAT",
    "RESULT_FORMAT",
    "ReceiptBindings",
    "load_judge_recipient_policy",
    "replay_signed_judge_verification_receipt",
    "verify_judge_evidence",
    "verify_judge_evidence_with_policy",
    "verify_signed_judge_verification_receipt",
    "verify_stored_judge_result",
    "write_signed_judge_verification_receipt",
]
