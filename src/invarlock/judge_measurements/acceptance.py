"""Independent offline acceptance of the bounded fixed-benchmark judge scope.

The externally supplied recipient policy is the authority. Embedded signer
keys are only evidence until their fingerprint and identity match that policy.
A receipt records this local computation; it is never accepted as a trust input.
Native receipts and recipient policies retain their existing narrower semantics.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from jsonschema import Draft202012Validator, ValidationError

from invarlock.captured_contracts import secure_directory
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_json import read_regular_file_bytes
from invarlock.judge_evidence_types import JudgeRecipientPolicy
from invarlock.judge_measurements.evidence import (
    DECISION_SCOPE,
    JudgeEvidenceError,
    object_sha256,
    read_object,
    replay_judge_evidence,
    signed_envelope_bytes,
)
from invarlock.judge_measurements.statistics import Decision
from invarlock.public_contracts import (
    load_judge_measurement_recipient_policy_schema,
    load_judge_measurement_verification_receipt_schema,
)

RECIPIENT_POLICY_FORMAT: Final = "invarlock/judge-measurement-recipient-policy-v1"
RECEIPT_FORMAT: Final = "invarlock/judge-measurement-verification-receipt-v1"


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
class JudgeVerificationReceipt:
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
    format: str = RECEIPT_FORMAT
    decision_scope: str = DECISION_SCOPE

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["errors"] = list(self.errors)
        return value


def _require_external(path: Path, evidence: Path, *, label: str) -> None:
    if path.resolve().is_relative_to(evidence.resolve()):
        raise JudgeEvidenceError(
            f"{label} must be supplied outside submitted judge evidence"
        )


def load_judge_recipient_policy(
    path: Path, *, evidence_path: Path
) -> JudgeRecipientPolicy:
    """Load the caller-selected external policy; never search evidence for trust."""
    path = Path(path).absolute()
    _require_external(path, Path(evidence_path), label="recipient policy")
    with secure_directory(path.parent):
        value = read_object(path, maximum=65536)
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
        value = read_regular_file_bytes(
            value, label="trusted signer key", max_bytes=65536
        )
    if len(value) == 32:
        return Ed25519PublicKey.from_public_bytes(value)
    key = serialization.load_pem_public_key(value)
    if not isinstance(key, Ed25519PublicKey):
        raise JudgeEvidenceError("trusted signer key must be Ed25519")
    return key


def verify_judge_evidence(
    path: Path,
    *,
    recipient_policy_path: Path,
    trusted_public_keys: Mapping[str, Path | bytes | Ed25519PublicKey] | None = None,
) -> JudgeVerificationReceipt:
    """Authenticate, replay, and decide from independently supplied recipient pins.

    A valid incomplete or advisory-only analysis may be authenticated and
    replayed, but cannot satisfy the required decision. No network is used.
    """
    root = Path(path).absolute()
    receipt = JudgeVerificationReceipt()
    try:
        policy = load_judge_recipient_policy(recipient_policy_path, evidence_path=root)
        receipt = replace(receipt, recipient_policy_sha256=object_sha256(policy))
        publication = replay_judge_evidence(root)
        envelope = publication.envelope
        receipt = replace(
            receipt,
            replayed=True,
            envelope_sha256=object_sha256(envelope),
            decision=publication.analysis_result.decision,
            intended_subject=envelope["intended_subject"],
            bindings=ReceiptBindings(**envelope["bindings"]),
        )
        signer = envelope["signer"]
        if signer is None or envelope["signature"] is None:
            raise JudgeEvidenceError(
                "unsigned judge evidence cannot be independently accepted"
            )
        receipt = replace(
            receipt,
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
        receipt = replace(receipt, authenticated=True)
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
        analysis = publication.analysis_result
        if analysis.policy.metric_name != policy["required_metric_name"]:
            raise JudgeEvidenceError(
                "judge metric differs from the recipient's required metric"
            )
        receipt = replace(receipt, verified=True)
        if not analysis.policy.required:
            raise JudgeEvidenceError(
                "advisory judge evidence cannot satisfy a required recipient decision"
            )
        if analysis.decision != "pass":
            raise JudgeEvidenceError(f"required judge decision is {analysis.decision}")
        receipt = replace(receipt, accepted=True)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        receipt = replace(
            receipt,
            accepted=False,
            errors=(str(exc)[:1000] or "judge evidence verification failed",),
        )
    # A locally produced receipt always satisfies its own closed public shape.
    Draft202012Validator(load_judge_measurement_verification_receipt_schema()).validate(
        receipt.to_dict()
    )
    return receipt


def verify_judge_evidence_with_policy(
    path: Path, recipient_policy_path: Path
) -> JudgeVerificationReceipt:
    return verify_judge_evidence(path, recipient_policy_path=recipient_policy_path)


def verify_stored_judge_receipt(
    receipt_path: Path,
    *,
    evidence_path: Path,
    recipient_policy_path: Path,
) -> JudgeVerificationReceipt:
    """Recompute a stored receipt; an asserted accepted flag is never authority."""
    actual = verify_judge_evidence_with_policy(evidence_path, recipient_policy_path)
    try:
        claimed = read_object(receipt_path, maximum=65536)
        Draft202012Validator(
            load_judge_measurement_verification_receipt_schema()
        ).validate(claimed)
        if object_sha256(claimed) != object_sha256(actual.to_dict()):
            raise JudgeEvidenceError(
                "stored judge receipt differs from independent recipient verification"
            )
    except (OSError, ValueError, TypeError, ValidationError) as exc:
        return replace(
            actual, accepted=False, verified=False, errors=(str(exc)[:1000],)
        )
    return actual
