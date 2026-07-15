"""Value contracts for runtime behavioral side and pair orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAliasType, cast

from invarlock.reporting.validation.runtime_behavioral_claim import (
    RuntimeBehavioralClaimVerificationResult,
)
from invarlock.reporting.validation.runtime_behavioral_observation import (
    RuntimeBehavioralMetricResult,
)
from invarlock.runtime_behavioral_claim_receipt import (
    RuntimeBehavioralClaimReceipt,
    RuntimeBehavioralEvidenceBindings,
)
from invarlock.runtime_provider_evidence import PersistedRuntimeProviderEvidence

RuntimeBehavioralRole = TypeAliasType(  # noqa: UP040
    "RuntimeBehavioralRole", Literal["baseline", "subject"]
)

RUNTIME_BEHAVIORAL_SIDE_REPORT_FORMAT = "invarlock/runtime-behavioral-side-report-v1"
RUNTIME_BEHAVIORAL_SIDE_CONFIG_FORMAT = "invarlock/runtime-behavioral-side-config-v1"
RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME = "evaluation.report.json"
RUNTIME_BEHAVIORAL_SIDE_CONFIG_FILENAME = "runtime-behavior.config.json"
MAX_RUNTIME_BEHAVIORAL_SIDE_FILE_BYTES = 64 * 1024 * 1024


class RuntimeBehaviorError(ValueError):
    """Raised when strict runtime behavioral evidence cannot be produced."""


@dataclass(frozen=True)
class RuntimeSideBundle:
    """A strictly reloaded side bundle and its portable evidence bindings."""

    role: RuntimeBehavioralRole
    directory: Path
    report_path: Path
    config_path: Path
    manifest_path: Path
    evidence: PersistedRuntimeProviderEvidence
    metric_result: RuntimeBehavioralMetricResult
    bindings: RuntimeBehavioralEvidenceBindings


@dataclass(frozen=True)
class RuntimePairVerification:
    """Successful paired replay and its atomically published positive receipt."""

    verification: RuntimeBehavioralClaimVerificationResult
    receipt: RuntimeBehavioralClaimReceipt
    receipt_path: Path


def require_role(value: object) -> RuntimeBehavioralRole:
    if value not in {"baseline", "subject"}:
        raise RuntimeBehaviorError("role must be baseline or subject")
    return cast(RuntimeBehavioralRole, value)


def report_payload(
    *,
    role: RuntimeBehavioralRole,
    provider_name: str,
    artifact_sha256: str,
    schedule_sha256: str,
    policy_digest: str,
    result: RuntimeBehavioralMetricResult,
) -> dict[str, object]:
    return {
        "format_version": RUNTIME_BEHAVIORAL_SIDE_REPORT_FORMAT,
        "role": role,
        "claim_set": "invarlock-runtime-behavioral-regression-v1",
        "metric": "exact_match",
        "verdict": "observation_verified",
        "provider_name": provider_name,
        "artifact_identity_sha256": artifact_sha256,
        "schedule_sha256": schedule_sha256,
        "policy_digest": policy_digest,
        "score": result.value,
        "correct_records": result.correct_records,
        "total_records": result.total_records,
        "aggregate_source_sha256": result.aggregate_source_sha256,
    }


def config_payload(
    *,
    role: RuntimeBehavioralRole,
    provider_name: str,
    artifact_sha256: str,
    schedule_sha256: str,
    policy_digest: str,
) -> dict[str, object]:
    return {
        "format_version": RUNTIME_BEHAVIORAL_SIDE_CONFIG_FORMAT,
        "role": role,
        "claim_set": "invarlock-runtime-behavioral-regression-v1",
        "metric": "exact_match",
        "provider_name": provider_name,
        "artifact_identity_sha256": artifact_sha256,
        "schedule_sha256": schedule_sha256,
        "policy_digest": policy_digest,
    }


def require_exact_payload(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    if dict(observed) != dict(expected):
        raise RuntimeBehaviorError(
            f"{label} does not match role, provider, artifact, schedule, and policy"
        )


__all__ = [
    "MAX_RUNTIME_BEHAVIORAL_SIDE_FILE_BYTES",
    "RUNTIME_BEHAVIORAL_SIDE_CONFIG_FILENAME",
    "RUNTIME_BEHAVIORAL_SIDE_CONFIG_FORMAT",
    "RUNTIME_BEHAVIORAL_SIDE_REPORT_FILENAME",
    "RUNTIME_BEHAVIORAL_SIDE_REPORT_FORMAT",
    "RuntimeBehavioralRole",
    "RuntimeBehaviorError",
    "RuntimePairVerification",
    "RuntimeSideBundle",
]
