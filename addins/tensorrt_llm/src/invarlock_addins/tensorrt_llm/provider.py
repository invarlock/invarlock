"""First-party TensorRT-LLM provider for authenticated engine bundles.

The provider is intentionally torch-free.  Backend imports and GPU execution live
behind the pinned runner protocol implemented by the TensorRT-LLM runtime image.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from invarlock_addins.tensorrt_llm import __version__ as ADDIN_VERSION
from invarlock_addins.tensorrt_llm.session import (
    TensorRTLLMRuntimeBindings,
    TensorRTLLMSession,
    TensorRTLLMSessionConfig,
    inspect_tensorrt_llm_inputs,
)

from invarlock.core.runtime_provider import (
    INVARLOCK_RUNTIME_PROVIDER_ABI,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeExecutionContext,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
    runtime_execution_settings_from_mapping,
)
from invarlock.core.runtime_provider.types import JSONScalar
from invarlock.runtime_providers.tensorrt_llm_identity import (
    read_tensorrt_llm_artifact_identity,
)
from invarlock.runtime_security_helpers import (
    RUNTIME_IMAGE_DIGEST_ENV,
    RUNTIME_IMAGE_ENV,
    strict_container_boundary_present,
)

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_COMPUTE_CAPABILITY = re.compile(r"^(0|[1-9][0-9]?)\.(0|[1-9][0-9]?)$")
_ALLOWED_SETTINGS = frozenset(
    {
        "backend_build_sha256",
        "backend_version",
        "batch_size",
        "builder_config_sha256",
        "context_length",
        "engine_bundle_tree_sha256",
        "engine_metadata_sha256",
        "file_inventory_sha256",
        "max_output_tokens",
        "runner_binary_sha256",
        "seed",
        "target_compute_capability",
        "timeout_seconds",
        "tokenizer_metadata_sha256",
    }
)
_REQUIRED_SETTINGS = _ALLOWED_SETTINGS
_DIGEST_SETTINGS = frozenset(
    {
        "backend_build_sha256",
        "builder_config_sha256",
        "engine_bundle_tree_sha256",
        "engine_metadata_sha256",
        "file_inventory_sha256",
        "runner_binary_sha256",
        "tokenizer_metadata_sha256",
    }
)
_POSITIVE_INTEGER_SETTINGS = frozenset(
    {"batch_size", "context_length", "max_output_tokens", "timeout_seconds"}
)


def _required_text(settings: Mapping[str, JSONScalar], name: str) -> str:
    value = settings.get(name)
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty trimmed printable string")
    return value


def _required_digest(settings: Mapping[str, JSONScalar], name: str) -> str:
    value = _required_text(settings, name)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return value


def _required_integer(
    settings: Mapping[str, JSONScalar], name: str, *, positive: bool
) -> int:
    value = settings.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        label = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {label} integer")
    if (positive and value <= 0) or (not positive and value < 0):
        label = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {label} integer")
    return value


def _require_strict_container_boundary(context: RuntimeExecutionContext) -> None:
    image_digest = context.container_image_digest
    if image_digest is None:
        raise ValueError(
            "tensorrt_llm execution requires a pinned outer container image"
        )
    if not strict_container_boundary_present():
        raise ValueError(
            "tensorrt_llm execution requires the authenticated container boundary"
        )
    observed_digest = os.environ.get(RUNTIME_IMAGE_DIGEST_ENV, "")
    if observed_digest != image_digest:
        raise ValueError(
            "tensorrt_llm runtime image digest does not match the container context"
        )
    image_ref = os.environ.get(RUNTIME_IMAGE_ENV, "")
    repository, separator, embedded_digest = image_ref.rpartition("@")
    if image_ref != observed_digest and not (
        repository and separator and embedded_digest == observed_digest
    ):
        raise ValueError(
            f"tensorrt_llm execution requires {RUNTIME_IMAGE_ENV} to embed the "
            "exact runtime image digest"
        )


class TensorRTLLMProvider:
    """Authenticate and execute one TensorRT-LLM engine through a pinned runner."""

    name = "tensorrt_llm"
    abi_version = INVARLOCK_RUNTIME_PROVIDER_ABI

    def validate_config(self, spec: ModelRuntimeSpec) -> None:
        if spec.provider_name != self.name:
            raise ValueError(
                f"provider_name must be {self.name!r}, got {spec.provider_name!r}"
            )
        unknown = set(spec.settings) - _ALLOWED_SETTINGS
        if unknown:
            rendered = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported tensorrt_llm setting(s): {rendered}")
        missing = _REQUIRED_SETTINGS - set(spec.settings)
        if missing:
            rendered = ", ".join(sorted(missing))
            raise ValueError(f"missing tensorrt_llm setting(s): {rendered}")
        for name in _DIGEST_SETTINGS:
            _required_digest(spec.settings, name)
        for name in _POSITIVE_INTEGER_SETTINGS:
            _required_integer(spec.settings, name, positive=True)
        _required_integer(spec.settings, "seed", positive=False)
        version = _required_text(spec.settings, "backend_version")
        if " ".join(version.split()) != version:
            raise ValueError("backend_version must use canonical single spacing")
        compute_capability = _required_text(spec.settings, "target_compute_capability")
        if _COMPUTE_CAPABILITY.fullmatch(compute_capability) is None:
            raise ValueError("target_compute_capability must use major.minor notation")
        expected_model_id = "tensorrt-llm-sha256-" + _required_digest(
            spec.settings, "engine_bundle_tree_sha256"
        )
        if spec.model_id != expected_model_id:
            raise ValueError(
                "tensorrt_llm model_id must be the privacy-safe full engine digest name"
            )

    def capabilities(self) -> RuntimeProviderCapabilities:
        return RuntimeProviderCapabilities(
            provider_name=self.name,
            artifact_formats=("tensorrt_llm_engine",),
            tasks=("text_causal",),
            metrics=("exact_match",),
            execution_modes=("container",),
            required_extra=None,
            required_image=None,
        )

    def inspect_runtime_spec(
        self,
        bindings: TensorRTLLMRuntimeBindings,
        *,
        seed: int,
        context_length: int,
        batch_size: int,
        max_output_tokens: int,
        timeout_seconds: int,
    ) -> ModelRuntimeSpec:
        """Derive one complete spec from authenticated local runtime inputs."""

        inspection = inspect_tensorrt_llm_inputs(bindings)
        if batch_size > inspection.engine_max_batch_size:
            raise ValueError("batch_size exceeds the engine build limit")
        if context_length > inspection.engine_max_input_len:
            raise ValueError("context_length exceeds the engine build limit")
        if context_length + max_output_tokens > inspection.engine_max_seq_len:
            raise ValueError(
                "context and output lengths exceed the engine sequence limit"
            )
        identity = inspection.artifact_identity
        spec = ModelRuntimeSpec(
            provider_name=self.name,
            model_id=identity.bundle_name,
            settings={
                "backend_build_sha256": inspection.backend_build_sha256,
                "backend_version": inspection.backend_version,
                "batch_size": batch_size,
                "builder_config_sha256": identity.builder_config_sha256,
                "context_length": context_length,
                "engine_bundle_tree_sha256": identity.engine_bundle_tree_sha256,
                "engine_metadata_sha256": identity.engine_metadata_sha256,
                "file_inventory_sha256": identity.file_inventory_sha256,
                "max_output_tokens": max_output_tokens,
                "runner_binary_sha256": inspection.runner_binary_sha256,
                "seed": seed,
                "target_compute_capability": identity.target_compute_capability,
                "timeout_seconds": timeout_seconds,
                "tokenizer_metadata_sha256": identity.tokenizer_metadata_sha256,
            },
        )
        self.validate_config(spec)
        if self.identify_artifact(spec) != identity:
            raise ValueError(
                "derived TensorRT-LLM settings do not reproduce artifact identity"
            )
        return spec

    def identify_artifact(self, spec: ModelRuntimeSpec) -> TensorRTLLMArtifactIdentity:
        self.validate_config(spec)
        return TensorRTLLMArtifactIdentity(
            bundle_name=spec.model_id,
            engine_bundle_tree_sha256=_required_digest(
                spec.settings, "engine_bundle_tree_sha256"
            ),
            file_inventory_sha256=_required_digest(
                spec.settings, "file_inventory_sha256"
            ),
            builder_config_sha256=_required_digest(
                spec.settings, "builder_config_sha256"
            ),
            tokenizer_metadata_sha256=_required_digest(
                spec.settings, "tokenizer_metadata_sha256"
            ),
            engine_metadata_sha256=_required_digest(
                spec.settings, "engine_metadata_sha256"
            ),
            target_compute_capability=_required_text(
                spec.settings, "target_compute_capability"
            ),
        )

    def prepare_execution(
        self,
        spec: ModelRuntimeSpec,
        resources: RuntimeArtifactResources,
    ) -> RuntimeExecutionContext:
        """Bind one root-confined engine bundle to its tokenizer and runner."""

        self.validate_config(spec)
        resources.require_support_names(
            frozenset({"runner_executable", "tokenizer_contract"})
        )
        if resources.device_kind != "cuda":
            raise ValueError("tensorrt_llm preparation requires a CUDA device")
        bindings = TensorRTLLMRuntimeBindings(
            engine_bundle_path=resources.primary_path(),
            tokenizer_contract_path=resources.support_path("tokenizer_contract"),
            runner_executable_path=resources.support_path("runner_executable"),
        )
        identity = self.identify_artifact(spec)
        observed_identity = read_tensorrt_llm_artifact_identity(
            bindings.engine_bundle_path,
            target_compute_capability=identity.target_compute_capability,
            tokenizer_metadata_sha256=identity.tokenizer_metadata_sha256,
        )
        if observed_identity != identity:
            raise ValueError(
                "primary TensorRT-LLM resource identity does not match spec"
            )
        return RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=resources.container_image_digest,
            device_kind="cuda",
            artifact_identity_sha256=artifact_identity_sha256(identity),
            provider_state=bindings,
        )

    def open(
        self,
        spec: ModelRuntimeSpec,
        context: RuntimeExecutionContext,
    ) -> TensorRTLLMSession:
        self.validate_config(spec)
        if not context.strict:
            raise ValueError("tensorrt_llm execution requires strict mode")
        if context.allow_network:
            raise ValueError("tensorrt_llm execution must disable network access")
        if context.container_image_digest is None:
            raise ValueError(
                "tensorrt_llm execution requires a pinned outer container image"
            )
        if context.artifact_identity_sha256 is None:
            raise ValueError(
                "tensorrt_llm execution requires artifact identity binding"
            )
        if context.device_kind != "cuda":
            raise ValueError("tensorrt_llm execution requires a CUDA device")
        if not isinstance(context.provider_state, TensorRTLLMRuntimeBindings):
            raise ValueError(
                "tensorrt_llm requires ephemeral runtime bindings in context"
            )
        if context.scorer is not None:
            raise ValueError("tensorrt_llm does not accept in-process scorer bindings")

        identity = self.identify_artifact(spec)
        expected_identity_sha256 = artifact_identity_sha256(identity)
        if context.artifact_identity_sha256 != expected_identity_sha256:
            raise ValueError(
                "runtime context artifact identity does not match tensorrt_llm spec"
            )
        observed_identity = read_tensorrt_llm_artifact_identity(
            context.provider_state.engine_bundle_path,
            target_compute_capability=identity.target_compute_capability,
            tokenizer_metadata_sha256=identity.tokenizer_metadata_sha256,
        )
        if observed_identity != identity:
            raise ValueError(
                "bound TensorRT-LLM artifact identity does not match the spec"
            )
        _require_strict_container_boundary(context)

        execution_settings = runtime_execution_settings_from_mapping(
            spec.settings,
            allow_network=False,
        )
        return TensorRTLLMSession(
            TensorRTLLMSessionConfig(
                artifact_identity=identity,
                backend_build_sha256=_required_digest(
                    spec.settings, "backend_build_sha256"
                ),
                backend_version=_required_text(spec.settings, "backend_version"),
                runner_binary_sha256=_required_digest(
                    spec.settings, "runner_binary_sha256"
                ),
                execution_settings=execution_settings,
                capabilities=self.capabilities(),
                plugin=RuntimeProviderPluginIdentity(
                    name=self.name,
                    distribution="invarlock-runtime-tensorrt-llm",
                    distribution_version=ADDIN_VERSION,
                ),
                outer_image_digest=context.container_image_digest,
                bindings=context.provider_state,
            )
        )


__all__ = ["TensorRTLLMProvider", "TensorRTLLMRuntimeBindings"]
