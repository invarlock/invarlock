"""Portable receipt for one strictly verified paired runtime-behavioral claim."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from jsonschema import Draft202012Validator

from invarlock.core.runtime_provider.claims import RUNTIME_BEHAVIORAL_CLAIM_SET
from invarlock.public_contracts import load_runtime_behavioral_claim_receipt_schema
from invarlock.reporting.validation.runtime_behavioral_claim import (
    RuntimeBehavioralClaimVerificationResult,
)

RUNTIME_BEHAVIORAL_CLAIM_RECEIPT_FORMAT = (
    "invarlock/runtime-behavioral-claim-receipt-v1"
)

type RuntimeBehavioralClaimVerdict = Literal["pass"]
type RuntimeBehavioralClaimMetric = Literal["exact_match"]

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_PREFIXED_SHA256 = re.compile(r"^sha256:[a-f0-9]{64}$")


class RuntimeBehavioralClaimReceiptError(ValueError):
    """Raised when a paired-claim receipt is incomplete or not authentic."""


def _require_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeBehavioralClaimReceiptError(
            f"{field_name} must be a lowercase sha256 digest"
        )
    return value


def _require_prefixed_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _PREFIXED_SHA256.fullmatch(value) is None:
        raise RuntimeBehavioralClaimReceiptError(
            f"{field_name} must be a canonical sha256-prefixed digest"
        )
    return value


def _require_score(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise RuntimeBehavioralClaimReceiptError(
            f"{field_name} must be a finite number in [0, 1]"
        )
    return float(value)


@dataclass(frozen=True)
class RuntimeBehavioralEvidenceBindings:
    """Digest-only bindings for one side of a portable paired claim."""

    runtime_manifest_sha256: str
    evaluation_report_sha256: str
    provider_receipt_sidecar_sha256: str
    scoring_observation_sidecar_sha256: str
    artifact_identity_sidecar_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_manifest_sha256",
            "evaluation_report_sha256",
            "provider_receipt_sidecar_sha256",
            "scoring_observation_sidecar_sha256",
            "artifact_identity_sidecar_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name=field_name)

    def to_payload(self) -> dict[str, str]:
        """Return the closed portable representation."""

        return {
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "evaluation_report_sha256": self.evaluation_report_sha256,
            "provider_receipt_sidecar_sha256": (self.provider_receipt_sidecar_sha256),
            "scoring_observation_sidecar_sha256": (
                self.scoring_observation_sidecar_sha256
            ),
            "artifact_identity_sidecar_sha256": (self.artifact_identity_sidecar_sha256),
        }


@dataclass(frozen=True)
class RuntimeBehavioralClaimReceipt:
    """Positive portable receipt produced from an independent strict replay."""

    baseline: RuntimeBehavioralEvidenceBindings
    subject: RuntimeBehavioralEvidenceBindings
    schedule_sha256: str
    policy_digest: str
    baseline_score: float
    subject_score: float
    regression: float
    format_version: str = field(
        default=RUNTIME_BEHAVIORAL_CLAIM_RECEIPT_FORMAT, init=False
    )
    claim_set: str = field(default=RUNTIME_BEHAVIORAL_CLAIM_SET, init=False)
    metric: RuntimeBehavioralClaimMetric = field(default="exact_match", init=False)
    verdict: RuntimeBehavioralClaimVerdict = field(default="pass", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, RuntimeBehavioralEvidenceBindings):
            raise RuntimeBehavioralClaimReceiptError(
                "baseline must be RuntimeBehavioralEvidenceBindings"
            )
        if not isinstance(self.subject, RuntimeBehavioralEvidenceBindings):
            raise RuntimeBehavioralClaimReceiptError(
                "subject must be RuntimeBehavioralEvidenceBindings"
            )
        _require_sha256(self.schedule_sha256, field_name="schedule_sha256")
        _require_prefixed_sha256(self.policy_digest, field_name="policy_digest")
        baseline_score = _require_score(
            self.baseline_score, field_name="baseline_score"
        )
        subject_score = _require_score(self.subject_score, field_name="subject_score")
        if (
            isinstance(self.regression, bool)
            or not isinstance(self.regression, int | float)
            or not math.isfinite(float(self.regression))
            or not -1.0 <= float(self.regression) <= 1.0
        ):
            raise RuntimeBehavioralClaimReceiptError(
                "regression must be a finite number in [-1, 1]"
            )
        expected_regression = baseline_score - subject_score
        if not math.isclose(
            float(self.regression),
            expected_regression,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeBehavioralClaimReceiptError(
                "regression must equal baseline_score - subject_score"
            )

    def to_payload(self) -> dict[str, object]:
        """Return the exact closed JSON object defined by the public contract."""

        return {
            "format_version": self.format_version,
            "claim_set": self.claim_set,
            "baseline": self.baseline.to_payload(),
            "subject": self.subject.to_payload(),
            "schedule_sha256": self.schedule_sha256,
            "policy_digest": self.policy_digest,
            "metric": self.metric,
            "baseline_score": self.baseline_score,
            "subject_score": self.subject_score,
            "regression": self.regression,
            "verdict": self.verdict,
        }


def _verified_replay_values(
    verification: RuntimeBehavioralClaimVerificationResult,
) -> tuple[float, float, float, str, str]:
    if not isinstance(verification, RuntimeBehavioralClaimVerificationResult):
        raise RuntimeBehavioralClaimReceiptError(
            "verification must be RuntimeBehavioralClaimVerificationResult"
        )
    if not verification.ok or verification.errors:
        raise RuntimeBehavioralClaimReceiptError(
            "claim receipt requires a successful independent strict replay"
        )
    if verification.claim_set != RUNTIME_BEHAVIORAL_CLAIM_SET:
        raise RuntimeBehavioralClaimReceiptError(
            "claim receipt requires the runtime behavioral claim set"
        )
    if verification.metric != "exact_match":
        raise RuntimeBehavioralClaimReceiptError(
            "claim receipt requires independently replayed exact_match"
        )
    baseline_score = _require_score(
        verification.baseline_score, field_name="verification.baseline_score"
    )
    subject_score = _require_score(
        verification.subject_score, field_name="verification.subject_score"
    )
    regression = verification.regression
    if (
        isinstance(regression, bool)
        or not isinstance(regression, int | float)
        or not math.isfinite(float(regression))
    ):
        raise RuntimeBehavioralClaimReceiptError(
            "verification.regression must be finite"
        )
    expected_regression = baseline_score - subject_score
    if not math.isclose(
        float(regression), expected_regression, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RuntimeBehavioralClaimReceiptError(
            "verification regression does not match replayed scores"
        )
    schedule_sha256 = _require_sha256(
        verification.schedule_sha256,
        field_name="verification.schedule_sha256",
    )
    policy_digest = _require_prefixed_sha256(
        verification.policy_digest,
        field_name="verification.policy_digest",
    )
    return (
        baseline_score,
        subject_score,
        expected_regression,
        schedule_sha256,
        policy_digest,
    )


def build_runtime_behavioral_claim_receipt(
    *,
    baseline: RuntimeBehavioralEvidenceBindings,
    subject: RuntimeBehavioralEvidenceBindings,
    verification: RuntimeBehavioralClaimVerificationResult,
) -> RuntimeBehavioralClaimReceipt:
    """Build a positive digest-only receipt from an independent paired replay."""

    baseline_score, subject_score, regression, schedule_sha256, policy_digest = (
        _verified_replay_values(verification)
    )
    return RuntimeBehavioralClaimReceipt(
        baseline=baseline,
        subject=subject,
        schedule_sha256=schedule_sha256,
        policy_digest=policy_digest,
        baseline_score=baseline_score,
        subject_score=subject_score,
        regression=regression,
    )


def canonical_runtime_behavioral_claim_receipt_json(
    receipt: RuntimeBehavioralClaimReceipt,
) -> bytes:
    """Serialize one receipt using the sole public canonical JSON convention."""

    if not isinstance(receipt, RuntimeBehavioralClaimReceipt):
        raise TypeError("receipt must be RuntimeBehavioralClaimReceipt")
    return json.dumps(
        receipt.to_payload(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def runtime_behavioral_claim_receipt_sha256(
    receipt: RuntimeBehavioralClaimReceipt,
) -> str:
    """Return the digest of the canonical portable receipt bytes."""

    return hashlib.sha256(
        canonical_runtime_behavioral_claim_receipt_json(receipt)
    ).hexdigest()


def _schema_error(payload: Mapping[str, object]) -> str | None:
    validator = Draft202012Validator(load_runtime_behavioral_claim_receipt_schema())
    errors = sorted(
        validator.iter_errors(dict(payload)),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return None
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    return f"claim receipt schema violation at {path}: {error.message}"


def verify_runtime_behavioral_claim_receipt(
    receipt: Mapping[str, object] | RuntimeBehavioralClaimReceipt,
    *,
    expected_baseline: RuntimeBehavioralEvidenceBindings,
    expected_subject: RuntimeBehavioralEvidenceBindings,
    expected_verification: RuntimeBehavioralClaimVerificationResult,
) -> RuntimeBehavioralClaimReceipt:
    """Verify schema and every binding against trusted independent replay inputs."""

    if isinstance(receipt, RuntimeBehavioralClaimReceipt):
        payload = receipt.to_payload()
    elif isinstance(receipt, Mapping):
        payload = dict(receipt)
    else:
        raise RuntimeBehavioralClaimReceiptError("claim receipt must be an object")
    schema_error = _schema_error(payload)
    if schema_error is not None:
        raise RuntimeBehavioralClaimReceiptError(schema_error)

    expected = build_runtime_behavioral_claim_receipt(
        baseline=expected_baseline,
        subject=expected_subject,
        verification=expected_verification,
    )
    if payload != expected.to_payload():
        raise RuntimeBehavioralClaimReceiptError(
            "claim receipt does not match the expected evidence and replay bindings"
        )
    return expected


__all__ = [
    "RUNTIME_BEHAVIORAL_CLAIM_RECEIPT_FORMAT",
    "RuntimeBehavioralClaimMetric",
    "RuntimeBehavioralClaimReceipt",
    "RuntimeBehavioralClaimReceiptError",
    "RuntimeBehavioralClaimVerdict",
    "RuntimeBehavioralEvidenceBindings",
    "build_runtime_behavioral_claim_receipt",
    "canonical_runtime_behavioral_claim_receipt_json",
    "runtime_behavioral_claim_receipt_sha256",
    "verify_runtime_behavioral_claim_receipt",
]
