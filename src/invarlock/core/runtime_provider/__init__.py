"""Torch-free runtime-provider protocol.

Runtime providers own immutable artifact identity, provider-local execution state,
and deterministic scoring. Provider internals are deliberately opaque to callers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from invarlock.output_text_contract import (
    ExactMatchOutputError,
    exact_match_output_text,
)

from .behavioral_observation import (
    RuntimeBehavioralMetric,
    RuntimeBehavioralMetricResult,
    RuntimeBehavioralObservationError,
    runtime_scoring_records_sha256,
    verify_runtime_behavioral_observation,
)
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
from .types import (
    RUNTIME_PROVIDER_ABI_VERSION,
    STANDARD_RUNTIME_TASKS,
    EvaluationBatch,
    EvaluationInputPart,
    EvaluationRecord,
    GGUFArtifactIdentity,
    HFSnapshotArtifactIdentity,
    ModelArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionContext,
    RuntimeExecutionSettings,
    RuntimeMetric,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    RuntimeTask,
    ScoringObservation,
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
    canonical_artifact_identity_json,
    canonical_evaluation_input_parts_json,
    evaluation_input_parts_sha256,
    require_runtime_task,
    runtime_execution_settings_from_mapping,
)

INVARLOCK_RUNTIME_PROVIDER_ABI = RUNTIME_PROVIDER_ABI_VERSION


@runtime_checkable
class RuntimeSession(Protocol):
    """An opened provider session with deterministic scoring and cleanup."""

    def score(self, batch: EvaluationBatch) -> ScoringObservation: ...

    def runtime_receipt(self) -> RuntimeProviderReceipt: ...

    def close(self) -> None: ...


@runtime_checkable
class RuntimeProvider(Protocol):
    """Stable provider ABI; implementations must keep backend imports lazy."""

    name: str
    abi_version: str

    def validate_config(self, spec: ModelRuntimeSpec) -> None: ...

    def capabilities(self) -> RuntimeProviderCapabilities: ...

    def identify_artifact(self, spec: ModelRuntimeSpec) -> ModelArtifactIdentity: ...

    def authenticate_artifact(
        self, spec: ModelRuntimeSpec, artifact_path: Path
    ) -> ModelArtifactIdentity: ...

    def prepare_execution(
        self, spec: ModelRuntimeSpec, resources: RuntimeArtifactResources
    ) -> RuntimeExecutionContext: ...

    def open(
        self, spec: ModelRuntimeSpec, context: RuntimeExecutionContext
    ) -> RuntimeSession: ...


@runtime_checkable
class RuntimeProviderInputPreflight(Protocol):
    """Optional model-free validation of authenticated schedule-bound inputs."""

    def validate_evaluation_inputs(
        self,
        spec: ModelRuntimeSpec,
        resources: RuntimeArtifactResources,
        schedule: RuntimeBehavioralSchedule,
    ) -> None: ...


def _runtime_input_preflight_hook(
    provider: RuntimeProvider,
) -> (
    Callable[
        [ModelRuntimeSpec, RuntimeArtifactResources, RuntimeBehavioralSchedule], object
    ]
    | None
):
    hook = getattr(provider, "validate_evaluation_inputs", None)
    if hook is None:
        return None
    if not callable(hook):
        raise TypeError("runtime provider input-preflight hook must be callable")
    return cast(
        Callable[
            [ModelRuntimeSpec, RuntimeArtifactResources, RuntimeBehavioralSchedule],
            object,
        ],
        hook,
    )


def validate_runtime_evaluation_inputs(
    provider: RuntimeProvider,
    spec: ModelRuntimeSpec,
    resources: RuntimeArtifactResources,
    schedule: RuntimeBehavioralSchedule,
) -> bool:
    """Invoke an optional provider input-preflight hook and report its presence."""

    hook = _runtime_input_preflight_hook(provider)
    if hook is None:
        return False
    result = hook(spec, resources, schedule)
    if result is not None:
        raise TypeError("runtime provider input-preflight hook must return None")
    return True


__all__ = [
    "STANDARD_RUNTIME_TASKS",
    "EvaluationBatch",
    "EvaluationInputPart",
    "EvaluationRecord",
    "ExactMatchOutputError",
    "GGUFArtifactIdentity",
    "HFSnapshotArtifactIdentity",
    "INVARLOCK_RUNTIME_PROVIDER_ABI",
    "ModelArtifactIdentity",
    "ModelRuntimeSpec",
    "RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT",
    "RUNTIME_PROVIDER_ABI_VERSION",
    "RuntimeBackendIdentity",
    "RuntimeArtifactResources",
    "RuntimeBehavioralDatasetIdentity",
    "RuntimeBehavioralMetric",
    "RuntimeBehavioralMetricResult",
    "RuntimeBehavioralObservationError",
    "RuntimeBehavioralSchedule",
    "RuntimeDeviceFacts",
    "RuntimeExecutionContext",
    "RuntimeExecutionSettings",
    "RuntimeMetric",
    "RuntimeTask",
    "RuntimeProvider",
    "RuntimeProviderInputPreflight",
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
    "canonical_evaluation_input_parts_json",
    "evaluation_input_parts_sha256",
    "exact_match_output_text",
    "runtime_execution_settings_from_mapping",
    "canonical_runtime_behavioral_schedule_json",
    "load_runtime_behavioral_schedule",
    "parse_runtime_behavioral_schedule_json",
    "runtime_scoring_records_sha256",
    "validate_runtime_evaluation_inputs",
    "require_runtime_task",
    "verify_runtime_behavioral_observation",
]
