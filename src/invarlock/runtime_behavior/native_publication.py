"""Project validated SDK results into native atomic runtime-side publication."""

from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from invarlock.core.runtime_provider import (
    ModelArtifactIdentity,
    RuntimeBehavioralSchedule,
    RuntimeProviderReceipt,
    ScoringObservation,
    canonical_runtime_behavioral_schedule_json,
)
from invarlock.runtime_provider_evidence import (
    PersistedRuntimeProviderEvidence,
    RuntimeProviderEvidencePaths,
)
from invarlock.runtime_security_helpers import RuntimeManifestExecution


def publish_runtime_side(
    *,
    role: str,
    output: Path,
    schedule: RuntimeBehavioralSchedule,
    policy_digest: str,
    artifact_identity: ModelArtifactIdentity,
    observation: ScoringObservation,
    receipt: RuntimeProviderReceipt,
    execution: RuntimeManifestExecution,
) -> PersistedRuntimeProviderEvidence:
    """Keep provider facts intact; Rust validates, binds, and publishes them."""
    from invarlock.native_backend import require_native, transact
    from invarlock.runtime_behavior.transaction import RuntimeEvidenceError

    native = require_native()
    observed_receipt = asdict(receipt)
    observed_observation = asdict(observation)
    config = {
        "directory": str(output),
        "role": role,
        "schedule_base64": base64.b64encode(
            canonical_runtime_behavioral_schedule_json(schedule)
        ).decode("ascii"),
        "policy_digest": policy_digest,
        "artifact_identity": asdict(artifact_identity),
        "records": observed_observation["records"],
        "plugin": observed_receipt["plugin"],
        "backend": observed_receipt["backend"],
        "capabilities": observed_receipt["capabilities"],
        "execution_settings": observed_receipt["execution_settings"],
        "device": observed_receipt["device"],
        "runtime_image_ref": execution.image_ref,
        "runtime_image_digest": execution.image_digest,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "observed_observation": observed_observation,
        "observed_receipt": observed_receipt,
    }
    try:
        result = transact("write-runtime-side", config)["payload"]
    except native.NativeVerificationError as exc:
        raise RuntimeEvidenceError(str(exc)) from exc
    material = result["material"]
    return PersistedRuntimeProviderEvidence(
        paths=RuntimeProviderEvidencePaths.in_directory(output),
        artifact_identity=artifact_identity,
        scoring_observation=observation,
        receipt=receipt,
        artifact_identity_bytes=base64.b64decode(
            material["model-artifact.identity.json"], validate=True
        ),
        scoring_observation_bytes=base64.b64decode(
            material["runtime-scoring.observation.json"], validate=True
        ),
        receipt_bytes=base64.b64decode(
            material["runtime-provider.receipt.json"], validate=True
        ),
        artifact_identity_sha256=result["artifact_identity_sha256"],
        scoring_observation_sha256=result["scoring_observation_sha256"],
        receipt_sha256=result["receipt_sha256"],
    )
