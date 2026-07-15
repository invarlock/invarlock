"""Torch-free runtime-provider protocol.

Runtime providers own immutable artifact identity and deterministic scoring. The
existing :class:`ModelAdapter` contract remains responsible for mutable model/edit
access when a provider can expose it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..api import ModelAdapter
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
    "RUNTIME_PROVIDER_ABI_VERSION",
    "RuntimeBackendIdentity",
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
    "canonical_artifact_identity_json",
]
