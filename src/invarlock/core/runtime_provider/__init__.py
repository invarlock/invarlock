"""Torch-free runtime-provider protocol.

Runtime providers own immutable artifact identity and deterministic scoring. The
existing :class:`ModelAdapter` contract remains responsible for mutable model/edit
access when a provider can expose it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..api import ModelAdapter
from .behavioral_schedule import (
    RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT,
    RuntimeBehavioralDatasetIdentity,
    RuntimeBehavioralSchedule,
    build_runtime_behavioral_schedule,
    build_runtime_behavioral_schedule_from_material,
    canonical_runtime_behavioral_schedule_json,
    load_runtime_behavioral_schedule,
    parse_runtime_behavioral_schedule_json,
)
from .claims import (
    RUNTIME_BEHAVIORAL_CLAIM_SET,
    RuntimeClaimCompatibility,
    evaluate_runtime_claim_compatibility,
    require_runtime_claim_compatibility,
)
from .types import (
    RUNTIME_PROVIDER_ABI_VERSION,
    EvaluationBatch,
    EvaluationRecord,
    GGUFArtifactIdentity,
    HFSnapshotArtifactIdentity,
    ModelArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionContext,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    ScoringObservation,
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
    canonical_artifact_identity_json,
    runtime_execution_settings_from_mapping,
)

INVARLOCK_RUNTIME_PROVIDER_ABI = RUNTIME_PROVIDER_ABI_VERSION


@runtime_checkable
class RuntimeSession(Protocol):
    """An opened provider session with deterministic scoring and cleanup."""

    def score(self, batch: EvaluationBatch) -> ScoringObservation: ...

    def runtime_receipt(self) -> RuntimeProviderReceipt: ...

    def model_adapter(self) -> ModelAdapter | None: ...

    def native_model(self) -> object | None: ...

    def close(self) -> None: ...


@runtime_checkable
class RuntimeProvider(Protocol):
    """Stable provider ABI; implementations must keep backend imports lazy."""

    name: str
    abi_version: str

    def validate_config(self, spec: ModelRuntimeSpec) -> None: ...

    def capabilities(self) -> RuntimeProviderCapabilities: ...

    def identify_artifact(self, spec: ModelRuntimeSpec) -> ModelArtifactIdentity: ...

    def open(
        self, spec: ModelRuntimeSpec, context: RuntimeExecutionContext
    ) -> RuntimeSession: ...


__all__ = [
    "EvaluationBatch",
    "EvaluationRecord",
    "GGUFArtifactIdentity",
    "HFSnapshotArtifactIdentity",
    "INVARLOCK_RUNTIME_PROVIDER_ABI",
    "ModelArtifactIdentity",
    "ModelRuntimeSpec",
    "RUNTIME_BEHAVIORAL_CLAIM_SET",
    "RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT",
    "RUNTIME_PROVIDER_ABI_VERSION",
    "RuntimeBackendIdentity",
    "RuntimeBehavioralDatasetIdentity",
    "RuntimeBehavioralSchedule",
    "RuntimeClaimCompatibility",
    "RuntimeDeviceFacts",
    "RuntimeExecutionContext",
    "RuntimeExecutionSettings",
    "RuntimeProvider",
    "RuntimeProviderCapabilities",
    "RuntimeProviderPluginIdentity",
    "RuntimeProviderReceipt",
    "RuntimeScoringRecord",
    "RuntimeSession",
    "ScoringObservation",
    "TensorRTLLMArtifactIdentity",
    "artifact_identity_sha256",
    "build_runtime_behavioral_schedule",
    "build_runtime_behavioral_schedule_from_material",
    "canonical_artifact_identity_json",
    "runtime_execution_settings_from_mapping",
    "canonical_runtime_behavioral_schedule_json",
    "evaluate_runtime_claim_compatibility",
    "load_runtime_behavioral_schedule",
    "parse_runtime_behavioral_schedule_json",
    "require_runtime_claim_compatibility",
]
