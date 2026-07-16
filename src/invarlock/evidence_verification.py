"""Single independent verification transaction for canonical evidence packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock.core.scorer_extension import ScorerExtensionRegistry
from invarlock.evidence_pack import verify_comparison_evidence
from invarlock.evidence_pack_json import StrictJsonError
from invarlock.evidence_pack_support import EvidencePackResult
from invarlock.evidence_receipt import (
    EvidenceReceiptError,
    write_signed_verification_receipt,
)


class EvidenceVerificationError(ValueError):
    """Raised when evidence is malformed, untrusted, or fails acceptance."""

    def __init__(
        self,
        message: str,
        *,
        exit_code: int = 2,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.payload = payload or {
            "format_version": "invarlock/evidence-verification-error-v1",
            "ok": False,
            "errors": [message],
            "warnings": [],
        }

    def as_json(self) -> str:
        return json.dumps(
            self.payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class EvidenceVerification:
    """Successful independent verification result."""

    evidence_path: Path
    payload: dict[str, Any]
    receipt_path: Path | None = None

    @property
    def summary(self) -> str:
        comparison = self.payload.get("comparison_id")
        signer = self.payload.get("signer_fingerprint")
        details = [f"Evidence: {self.evidence_path}"]
        if isinstance(comparison, str):
            details.append(f"Comparison: {comparison}")
        if isinstance(signer, str):
            details.append(f"Evidence signer: {signer}")
        verifier = self.payload.get("verifier_fingerprint")
        if isinstance(verifier, str):
            details.append(f"Verifier signer: {verifier}")
        if self.receipt_path is not None:
            details.append(f"Receipt: {self.receipt_path}")
        return "\n".join(details)

    def as_json(self) -> str:
        return json.dumps(
            self.payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _require_file(path: Path | None, *, label: str) -> Path:
    if path is None:
        raise EvidenceVerificationError(f"{label} is required")
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise EvidenceVerificationError(f"{label} must be a real regular file")
    return candidate


def _failed(result: EvidencePackResult, payload: dict[str, Any]) -> None:
    if bool(payload.get("ok")):
        return
    errors = payload.get("errors")
    message = (
        "; ".join(str(item) for item in errors)
        if isinstance(errors, list) and errors
        else "evidence verification failed"
    )
    exit_code = int(result.status) or 1
    raise EvidenceVerificationError(message, exit_code=exit_code, payload=payload)


def verify_evidence(
    evidence_path: Path,
    *,
    policy_path: Path | None,
    expected_baseline_artifact: str | None,
    expected_subject_artifact: str | None,
    expected_schedule: str | None,
    expected_baseline_runtime: str | None,
    expected_subject_runtime: str | None,
    expected_signer: str | None,
    receipt_path: Path | None = None,
    verifier_signing_key_path: Path | None = None,
    verifier_identity: str | None = None,
    scorer_registry: ScorerExtensionRegistry | None = None,
) -> EvidenceVerification:
    """Verify one pack with roots that cannot be selected by that pack.

    The canonical evidence pack requires independent per-side runtime roots,
    an evidence signer root, and a distinct verifier key/identity that signs a
    receipt outside the immutable pack.
    """

    evidence = Path(evidence_path)
    if not evidence.is_dir() or evidence.is_symlink():
        raise EvidenceVerificationError("evidence must be a real directory")
    policy = _require_file(policy_path, label="independent policy")
    if (
        not isinstance(expected_baseline_artifact, str)
        or not expected_baseline_artifact
    ):
        raise EvidenceVerificationError(
            "independent baseline artifact anchor is required"
        )
    if not isinstance(expected_subject_artifact, str) or not expected_subject_artifact:
        raise EvidenceVerificationError(
            "independent subject artifact anchor is required"
        )
    if not isinstance(expected_schedule, str) or not expected_schedule:
        raise EvidenceVerificationError("independent schedule anchor is required")
    if not isinstance(expected_baseline_runtime, str) or not expected_baseline_runtime:
        raise EvidenceVerificationError(
            "independent baseline runtime anchor is required"
        )
    if not isinstance(expected_subject_runtime, str) or not expected_subject_runtime:
        raise EvidenceVerificationError(
            "independent subject runtime anchor is required"
        )
    if not isinstance(expected_signer, str) or not expected_signer:
        raise EvidenceVerificationError("independent evidence signer is required")
    runtimes = {
        "baseline": expected_baseline_runtime,
        "subject": expected_subject_runtime,
    }
    artifacts = {
        "baseline": expected_baseline_artifact,
        "subject": expected_subject_artifact,
    }
    receipt = Path(receipt_path) if receipt_path is not None else None
    if receipt is None:
        raise EvidenceVerificationError(
            "signed verification receipt destination is required for evidence-pack-v1"
        )
    verifier_key = _require_file(
        verifier_signing_key_path, label="verifier Ed25519 signing key"
    )
    if not isinstance(verifier_identity, str) or not verifier_identity.strip():
        raise EvidenceVerificationError("verifier identity is required")
    try:
        receipt.resolve().relative_to(evidence.resolve())
    except ValueError:
        pass
    else:
        raise EvidenceVerificationError(
            "verification receipt must remain outside the immutable evidence pack"
        )

    result = verify_comparison_evidence(
        evidence,
        policy_path=policy,
        expected_artifact_digests=artifacts,
        expected_schedule_digest=expected_schedule,
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=expected_signer,
        scorer_registry=scorer_registry,
    )
    payload = dict(result.payload)
    try:
        verifier_fingerprint = write_signed_verification_receipt(
            evidence,
            result,
            receipt,
            policy_path=policy,
            expected_artifact_digests=artifacts,
            expected_schedule_digest=expected_schedule,
            expected_runtime_digests=runtimes,
            expected_pack_signer_fingerprint=expected_signer,
            verifier_identity=verifier_identity,
            verifier_signing_key_path=verifier_key,
        )
    except (EvidenceReceiptError, StrictJsonError) as exc:
        raise EvidenceVerificationError(str(exc)) from exc
    payload["signed_receipt"] = str(receipt.resolve())
    payload["verifier_identity"] = verifier_identity
    payload["verifier_fingerprint"] = verifier_fingerprint
    _failed(result, payload)
    return EvidenceVerification(evidence.resolve(), payload, receipt.resolve())


__all__ = [
    "EvidenceVerification",
    "EvidenceVerificationError",
    "verify_evidence",
]
