"""Hugging Face reference runtime provider.

This provider deliberately wraps the adapter, model, and scorer already resolved by
the established execution path. It does not load a model or import an optional
backend; that preserves the current adapter ABI and avoids a second model load.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from invarlock.core.api import ModelAdapter
from invarlock.core.runtime_provider import (
    INVARLOCK_RUNTIME_PROVIDER_ABI,
    EvaluationBatch,
    HFSnapshotArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeExecutionContext,
    RuntimeProviderCapabilities,
    RuntimeProviderReceipt,
    ScoringObservation,
    artifact_identity_sha256,
)
from invarlock.core.runtime_provider.types import JSONScalar, RuntimeScorer

_ALLOWED_SETTINGS = frozenset(
    {
        "batch_size",
        "checkpoint_tree_sha256",
        "context_length",
        "immutable_revision",
        "max_output_tokens",
        "offline",
        "seed",
        "timeout_seconds",
        "tokenizer_metadata_sha256",
    }
)
_POSITIVE_INTEGER_SETTINGS = frozenset(
    {"batch_size", "context_length", "max_output_tokens", "timeout_seconds"}
)
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _optional_text(settings: Mapping[str, JSONScalar], name: str) -> str | None:
    value = settings.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _optional_sha256(settings: Mapping[str, JSONScalar], name: str) -> str | None:
    value = _optional_text(settings, name)
    if value is None:
        return None
    canonical = value.removeprefix("sha256:")
    if _SHA256.fullmatch(canonical) is None:
        raise ValueError(f"{name} must be a sha256 digest")
    return canonical


def _is_local_path_like(model_id: str) -> bool:
    try:
        exists_locally = Path(model_id).exists()
    except OSError:
        # Path APIs reject some malformed/oversized host inputs. Treat those as
        # path-like so they can never flow into a public artifact identity.
        exists_locally = True
    return bool(
        Path(model_id).is_absolute()
        or exists_locally
        or model_id.startswith(("./", "../", "~/"))
        or "\\" in model_id
        or _WINDOWS_ABSOLUTE_PATH.match(model_id)
    )


def _validate_setting_values(spec: ModelRuntimeSpec) -> None:
    for name in _POSITIVE_INTEGER_SETTINGS:
        value = spec.settings.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")

    seed = spec.settings.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")

    offline = spec.settings.get("offline")
    if offline is not None and not isinstance(offline, bool):
        raise ValueError("offline must be boolean")

    _optional_text(spec.settings, "immutable_revision")
    _optional_sha256(spec.settings, "checkpoint_tree_sha256")
    _optional_sha256(spec.settings, "tokenizer_metadata_sha256")


@dataclass
class _HFTransformersSession:
    _adapter: ModelAdapter
    _model: object
    _scorer: RuntimeScorer
    _close_callback: Callable[[], None] | None = None
    _artifact_identity_sha256: str | None = None
    _closed: bool = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime provider session is closed")

    def score(self, batch: EvaluationBatch) -> ScoringObservation:
        """Delegate scoring and fail closed on schedule/pairing drift."""

        self._require_open()
        observation = self._scorer(batch)
        if not isinstance(observation, ScoringObservation):
            raise TypeError("runtime scorer must return ScoringObservation")
        if observation.provider_name != HFTransformersProvider.name:
            raise ValueError("scoring observation provider does not match session")
        if (
            self._artifact_identity_sha256 is not None
            and observation.artifact_identity_sha256 != self._artifact_identity_sha256
        ):
            raise ValueError(
                "scoring observation artifact identity does not match session"
            )
        if observation.schedule_sha256 != batch.schedule_sha256:
            raise ValueError("scoring observation schedule does not match batch")
        expected_pairing = tuple(
            (record.record_id, record.input_sha256) for record in batch.records
        )
        observed_pairing = tuple(
            (record.record_id, record.input_sha256) for record in observation.records
        )
        if observed_pairing != expected_pairing:
            raise ValueError("scoring observation pairing does not match batch")
        return observation

    def runtime_receipt(self) -> RuntimeProviderReceipt:
        """Require the receipt pipeline to bind real backend and digest facts."""

        self._require_open()
        raise RuntimeError(
            "runtime provider receipt is unavailable until provenance facts are bound"
        )

    def model_adapter(self) -> ModelAdapter:
        """Return the exact adapter selected by the existing HF execution path."""

        self._require_open()
        return self._adapter

    def native_model(self) -> object:
        """Return the exact already-loaded model; never create a second instance."""

        self._require_open()
        return self._model

    def close(self) -> None:
        """Run the existing lifecycle callback at most once."""

        if self._closed:
            return
        self._closed = True
        if self._close_callback is not None:
            self._close_callback()


class HFTransformersProvider:
    """Reference provider for the existing in-process HF adapter pipeline."""

    name = "hf_transformers"
    abi_version = INVARLOCK_RUNTIME_PROVIDER_ABI

    def validate_config(self, spec: ModelRuntimeSpec) -> None:
        if spec.provider_name != self.name:
            raise ValueError(
                f"provider_name must be {self.name!r}, got {spec.provider_name!r}"
            )
        adapter_name = spec.adapter_name
        if adapter_name is not None and not (
            adapter_name in {"auto", "auto_hf"} or adapter_name.startswith("hf_")
        ):
            raise ValueError(
                "hf_transformers adapter_name must be auto, auto_hf, or hf_*"
            )
        unknown = set(spec.settings) - _ALLOWED_SETTINGS
        if unknown:
            rendered = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported hf_transformers setting(s): {rendered}")
        _validate_setting_values(spec)

    def capabilities(self) -> RuntimeProviderCapabilities:
        return RuntimeProviderCapabilities(
            provider_name=self.name,
            artifact_formats=("hf_snapshot",),
            tasks=("text_causal",),
            metrics=("exact_match", "multiple_choice_accuracy"),
            execution_modes=("in_process",),
            required_extra="hf",
            required_image=None,
            platform_constraints=("python",),
            evidence_surfaces=(
                "behavior",
                "tokenizer",
                "weights",
                "modules",
                "activations",
            ),
            supported_claim_sets=(
                "invarlock-weight-edit-regression-v2",
                "invarlock-runtime-behavioral-regression-v1",
            ),
        )

    def identify_artifact(self, spec: ModelRuntimeSpec) -> HFSnapshotArtifactIdentity:
        self.validate_config(spec)
        immutable_revision = _optional_text(spec.settings, "immutable_revision")
        checkpoint_tree_sha256 = _optional_sha256(
            spec.settings, "checkpoint_tree_sha256"
        )
        tokenizer_metadata_sha256 = _optional_sha256(
            spec.settings, "tokenizer_metadata_sha256"
        )
        if immutable_revision is None and checkpoint_tree_sha256 is None:
            raise ValueError(
                "hf_transformers requires an immutable identity revision or tree digest"
            )
        if tokenizer_metadata_sha256 is None:
            raise ValueError(
                "hf_transformers requires tokenizer_metadata_sha256 for artifact identity"
            )
        logical_model_id = spec.model_id
        if _is_local_path_like(spec.model_id):
            if checkpoint_tree_sha256 is None:
                raise ValueError(
                    "local hf_transformers paths require checkpoint_tree_sha256"
                )
            logical_model_id = f"local-checkpoint-{checkpoint_tree_sha256[:12]}"
        return HFSnapshotArtifactIdentity(
            model_id=logical_model_id,
            immutable_revision=immutable_revision,
            checkpoint_tree_sha256=checkpoint_tree_sha256,
            tokenizer_metadata_sha256=tokenizer_metadata_sha256,
        )

    def open(
        self,
        spec: ModelRuntimeSpec,
        context: RuntimeExecutionContext,
    ) -> _HFTransformersSession:
        self.validate_config(spec)
        if context.strict and context.allow_network:
            raise ValueError("strict hf_transformers execution must disable network")
        if context.strict and context.artifact_identity_sha256 is None:
            raise ValueError(
                "strict hf_transformers execution requires artifact_identity_sha256"
            )
        for field_name in ("model_adapter", "native_model", "scorer"):
            if getattr(context, field_name) is None:
                raise ValueError(
                    f"hf_transformers requires prebound {field_name} in context"
                )
        if context.artifact_identity_sha256 is not None:
            expected_identity_sha256 = artifact_identity_sha256(
                self.identify_artifact(spec)
            )
            if context.artifact_identity_sha256 != expected_identity_sha256:
                raise ValueError(
                    "runtime context artifact identity does not match model spec"
                )
        return _HFTransformersSession(
            _adapter=cast(ModelAdapter, context.model_adapter),
            _model=context.native_model,
            _scorer=cast(RuntimeScorer, context.scorer),
            _close_callback=context.close_callback,
            _artifact_identity_sha256=context.artifact_identity_sha256,
        )


@dataclass(frozen=True)
class HFTransformersSessionFactory:
    """Bind the existing HF load once and accept the scorer at its later boundary.

    Core orchestration resolves and loads the adapter/model before its guarded scorer
    exists. This factory is the narrow hand-off between those phases: it stores the
    exact objects and never calls an adapter loader. Once the scorer is available,
    ``open`` creates the provider session without duplicating model state.
    """

    spec: ModelRuntimeSpec
    authenticated_artifact_identity: HFSnapshotArtifactIdentity
    model_adapter: object = field(repr=False, compare=False)
    native_model: object = field(repr=False, compare=False)
    strict: bool
    allow_network: bool
    container_image_digest: str | None
    device_kind: str
    artifact_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        provider = HFTransformersProvider()
        provider.validate_config(self.spec)
        declared_identity = provider.identify_artifact(self.spec)
        if self.authenticated_artifact_identity != declared_identity:
            raise ValueError(
                "authenticated artifact identity does not match the runtime spec"
            )
        identity_sha256 = artifact_identity_sha256(self.authenticated_artifact_identity)
        object.__setattr__(self, "artifact_identity_sha256", identity_sha256)

    def open(
        self,
        scorer: RuntimeScorer,
        *,
        close_callback: Callable[[], None] | None = None,
    ) -> _HFTransformersSession:
        """Open over an injected scorer and the exact prebound adapter/model."""

        return HFTransformersProvider().open(
            self.spec,
            RuntimeExecutionContext(
                strict=self.strict,
                allow_network=self.allow_network,
                container_image_digest=self.container_image_digest,
                device_kind=self.device_kind,
                artifact_identity_sha256=self.artifact_identity_sha256,
                model_adapter=self.model_adapter,
                native_model=self.native_model,
                scorer=scorer,
                close_callback=close_callback,
            ),
        )


__all__ = [
    "HFTransformersProvider",
    "HFTransformersSessionFactory",
    "INVARLOCK_RUNTIME_PROVIDER_ABI",
]
