"""Canonical persistence for runtime-provider evidence sidecars.

Providers supply measured facts and provenance.  This module owns their portable
JSON encoding, strict reload, and structural cross-binding; behavioral aggregates
remain the responsibility of the independent reporting verifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from invarlock.core.runtime_provider import (
    GGUFArtifactIdentity,
    HFSnapshotArtifactIdentity,
    ModelArtifactIdentity,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    ScoringObservation,
    TensorRTLLMArtifactIdentity,
    artifact_identity_sha256,
)
from invarlock.core.runtime_provider.types import (
    ArtifactFormat,
    EvidenceSurface,
    RuntimeExecutionMode,
    RuntimeMetric,
    RuntimeTask,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.public_contracts import (
    load_model_artifact_identity_schema,
    load_runtime_provider_capabilities_schema,
    load_runtime_provider_receipt_schema,
    load_runtime_scoring_observation_schema,
)

ARTIFACT_IDENTITY_FILENAME = "model-artifact.identity.json"
SCORING_OBSERVATION_FILENAME = "runtime-scoring.observation.json"
PROVIDER_RECEIPT_FILENAME = "runtime-provider.receipt.json"
MAX_RUNTIME_PROVIDER_SIDECAR_BYTES = 64 * 1024 * 1024

type RuntimeProviderEvidenceValue = (
    ModelArtifactIdentity
    | RuntimeProviderCapabilities
    | RuntimeProviderReceipt
    | ScoringObservation
)

_IMAGE_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")


class RuntimeProviderEvidenceError(ValueError):
    """Raised when persisted provider evidence is malformed or inconsistently bound."""


@dataclass(frozen=True)
class RuntimeProviderEvidencePaths:
    """The three sibling files required by runtime manifest v2."""

    artifact_identity: Path
    scoring_observation: Path
    receipt: Path

    @classmethod
    def in_directory(cls, directory: str | Path) -> RuntimeProviderEvidencePaths:
        root = Path(directory)
        return cls(
            artifact_identity=root / ARTIFACT_IDENTITY_FILENAME,
            scoring_observation=root / SCORING_OBSERVATION_FILENAME,
            receipt=root / PROVIDER_RECEIPT_FILENAME,
        )


@dataclass(frozen=True)
class PersistedRuntimeProviderEvidence:
    """One strictly reloaded, typed, and cross-bound sidecar bundle."""

    paths: RuntimeProviderEvidencePaths
    artifact_identity: ModelArtifactIdentity
    scoring_observation: ScoringObservation
    receipt: RuntimeProviderReceipt
    artifact_identity_bytes: bytes
    scoring_observation_bytes: bytes
    receipt_bytes: bytes
    artifact_identity_sha256: str
    scoring_observation_sha256: str
    receipt_sha256: str

    @property
    def capabilities(self) -> RuntimeProviderCapabilities:
        return self.receipt.capabilities


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: RuntimeProviderEvidenceValue) -> bytes:
    try:
        return json.dumps(
            asdict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeProviderEvidenceError(
            "runtime provider evidence is not finite canonical JSON"
        ) from exc


def _validated_object(
    payload: bytes,
    *,
    label: str,
    schema: Mapping[str, object],
) -> dict[str, object]:
    if len(payload) > MAX_RUNTIME_PROVIDER_SIDECAR_BYTES:
        raise RuntimeProviderEvidenceError(
            f"{label} exceeds the {MAX_RUNTIME_PROVIDER_SIDECAR_BYTES}-byte size limit"
        )
    try:
        decoded = parse_json_bytes(payload, label=label)
    except StrictJsonError as exc:
        raise RuntimeProviderEvidenceError(str(exc)) from exc
    if not isinstance(decoded, dict):
        raise RuntimeProviderEvidenceError(f"{label} must be a JSON object")
    validator = Draft202012Validator(dict(schema))
    errors = sorted(
        validator.iter_errors(decoded),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise RuntimeProviderEvidenceError(
            f"{label} schema validation failed at {path}: {error.message}"
        )
    return cast(dict[str, object], decoded)


def _reconstruct[T](
    payload: bytes,
    *,
    label: str,
    schema: Mapping[str, object],
    builder: Callable[[dict[str, object]], T],
) -> T:
    decoded = _validated_object(payload, label=label, schema=schema)
    try:
        return builder(decoded)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeProviderEvidenceError(
            f"{label} could not be reconstructed: {exc}"
        ) from exc


def _optional_text(value: object) -> str | None:
    return cast(str | None, value)


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(cast(list[str], value))


def _artifact_from_payload(payload: dict[str, object]) -> ModelArtifactIdentity:
    artifact_format = payload["artifact_format"]
    if artifact_format == "hf_snapshot":
        return HFSnapshotArtifactIdentity(
            model_id=cast(str, payload["model_id"]),
            immutable_revision=_optional_text(payload["immutable_revision"]),
            checkpoint_tree_sha256=_optional_text(payload["checkpoint_tree_sha256"]),
            tokenizer_metadata_sha256=cast(str, payload["tokenizer_metadata_sha256"]),
        )
    if artifact_format == "gguf":
        return GGUFArtifactIdentity(
            artifact_name=cast(str, payload["artifact_name"]),
            sha256=cast(str, payload["sha256"]),
            byte_length=cast(int, payload["byte_length"]),
            gguf_metadata_sha256=cast(str, payload["gguf_metadata_sha256"]),
            tensor_inventory_sha256=cast(str, payload["tensor_inventory_sha256"]),
            tokenizer_metadata_sha256=cast(str, payload["tokenizer_metadata_sha256"]),
        )
    if artifact_format == "tensorrt_llm_engine":
        return TensorRTLLMArtifactIdentity(
            bundle_name=cast(str, payload["bundle_name"]),
            engine_bundle_tree_sha256=cast(str, payload["engine_bundle_tree_sha256"]),
            file_inventory_sha256=cast(str, payload["file_inventory_sha256"]),
            builder_config_sha256=cast(str, payload["builder_config_sha256"]),
            tokenizer_metadata_sha256=cast(str, payload["tokenizer_metadata_sha256"]),
            engine_metadata_sha256=cast(str, payload["engine_metadata_sha256"]),
            target_compute_capability=cast(str, payload["target_compute_capability"]),
        )
    raise RuntimeProviderEvidenceError(
        f"unsupported artifact identity format: {artifact_format!r}"
    )


def _capabilities_from_payload(
    payload: dict[str, object],
) -> RuntimeProviderCapabilities:
    return RuntimeProviderCapabilities(
        provider_name=cast(str, payload["provider_name"]),
        artifact_formats=cast(
            tuple[ArtifactFormat, ...], _string_tuple(payload["artifact_formats"])
        ),
        tasks=cast(tuple[RuntimeTask, ...], _string_tuple(payload["tasks"])),
        metrics=cast(tuple[RuntimeMetric, ...], _string_tuple(payload["metrics"])),
        execution_modes=cast(
            tuple[RuntimeExecutionMode, ...],
            _string_tuple(payload["execution_modes"]),
        ),
        required_extra=_optional_text(payload["required_extra"]),
        required_image=_optional_text(payload["required_image"]),
        platform_constraints=_string_tuple(payload["platform_constraints"]),
        evidence_surfaces=cast(
            tuple[EvidenceSurface, ...],
            _string_tuple(payload["evidence_surfaces"]),
        ),
        supported_claim_sets=_string_tuple(payload["supported_claim_sets"]),
        degraded_modes=_string_tuple(payload["degraded_modes"]),
        unavailable_modes=_string_tuple(payload["unavailable_modes"]),
    )


def _scoring_record_from_payload(payload: Mapping[str, object]) -> RuntimeScoringRecord:
    return RuntimeScoringRecord(
        record_id=cast(str, payload["record_id"]),
        input_sha256=cast(str, payload["input_sha256"]),
        status=cast(Any, payload["status"]),
        output_text=_optional_text(payload["output_text"]),
        output_sha256=_optional_text(payload["output_sha256"]),
        logprob_sum=cast(int | float | None, payload["logprob_sum"]),
        token_count=cast(int | None, payload["token_count"]),
        utf8_byte_count=cast(int | None, payload["utf8_byte_count"]),
        error_code=_optional_text(payload["error_code"]),
    )


def _observation_from_payload(payload: dict[str, object]) -> ScoringObservation:
    records = cast(list[Mapping[str, object]], payload["records"])
    return ScoringObservation(
        provider_name=cast(str, payload["provider_name"]),
        artifact_identity_sha256=cast(str, payload["artifact_identity_sha256"]),
        schedule_sha256=cast(str, payload["schedule_sha256"]),
        records=tuple(_scoring_record_from_payload(record) for record in records),
        aggregate_source_sha256=cast(str, payload["aggregate_source_sha256"]),
    )


def _receipt_from_payload(payload: dict[str, object]) -> RuntimeProviderReceipt:
    plugin = cast(dict[str, object], payload["plugin"])
    backend = cast(dict[str, object], payload["backend"])
    execution = cast(dict[str, object], payload["execution_settings"])
    device = cast(dict[str, object], payload["device"])
    return RuntimeProviderReceipt(
        plugin=RuntimeProviderPluginIdentity(
            name=cast(str, plugin["name"]),
            distribution=cast(str, plugin["distribution"]),
            distribution_version=cast(str, plugin["distribution_version"]),
        ),
        backend=RuntimeBackendIdentity(
            name=cast(str, backend["name"]),
            version=cast(str, backend["version"]),
            source_sha256=_optional_text(backend["source_sha256"]),
            binary_sha256=_optional_text(backend["binary_sha256"]),
            build_sha256=_optional_text(backend["build_sha256"]),
        ),
        capabilities=_capabilities_from_payload(
            cast(dict[str, object], payload["capabilities"])
        ),
        artifact_identity=_artifact_from_payload(
            cast(dict[str, object], payload["artifact_identity"])
        ),
        execution_settings=RuntimeExecutionSettings(
            seed=cast(int, execution["seed"]),
            context_length=cast(int, execution["context_length"]),
            batch_size=cast(int, execution["batch_size"]),
            max_output_tokens=cast(int, execution["max_output_tokens"]),
            timeout_seconds=cast(int, execution["timeout_seconds"]),
            allow_network=cast(bool, execution["allow_network"]),
        ),
        device=RuntimeDeviceFacts(
            device_kind=cast(str, device["device_kind"]),
            device_name=cast(str, device["device_name"]),
            compute_capability=_optional_text(device["compute_capability"]),
            driver_version=_optional_text(device["driver_version"]),
            cuda_runtime_version=_optional_text(device["cuda_runtime_version"]),
        ),
        outer_image_digest=_optional_text(payload["outer_image_digest"]),
        scoring_observation_sha256=cast(str, payload["scoring_observation_sha256"]),
    )


def encode_artifact_identity(identity: ModelArtifactIdentity) -> bytes:
    encoded = _canonical_json_bytes(identity)
    decode_artifact_identity(encoded)
    return encoded


def decode_artifact_identity(payload: bytes) -> ModelArtifactIdentity:
    return _reconstruct(
        payload,
        label="model artifact identity",
        schema=load_model_artifact_identity_schema(),
        builder=_artifact_from_payload,
    )


def encode_runtime_provider_capabilities(
    capabilities: RuntimeProviderCapabilities,
) -> bytes:
    encoded = _canonical_json_bytes(capabilities)
    decode_runtime_provider_capabilities(encoded)
    return encoded


def decode_runtime_provider_capabilities(
    payload: bytes,
) -> RuntimeProviderCapabilities:
    return _reconstruct(
        payload,
        label="runtime provider capabilities",
        schema=load_runtime_provider_capabilities_schema(),
        builder=_capabilities_from_payload,
    )


def encode_scoring_observation(observation: ScoringObservation) -> bytes:
    encoded = _canonical_json_bytes(observation)
    decode_scoring_observation(encoded)
    return encoded


def decode_scoring_observation(payload: bytes) -> ScoringObservation:
    return _reconstruct(
        payload,
        label="runtime scoring observation",
        schema=load_runtime_scoring_observation_schema(),
        builder=_observation_from_payload,
    )


def encode_runtime_provider_receipt(receipt: RuntimeProviderReceipt) -> bytes:
    encoded = _canonical_json_bytes(receipt)
    decode_runtime_provider_receipt(encoded)
    return encoded


def decode_runtime_provider_receipt(payload: bytes) -> RuntimeProviderReceipt:
    return _reconstruct(
        payload,
        label="runtime provider receipt",
        schema=load_runtime_provider_receipt_schema(),
        builder=_receipt_from_payload,
    )


def runtime_provider_evidence_errors(
    *,
    artifact_identity: ModelArtifactIdentity,
    scoring_observation: ScoringObservation,
    receipt: RuntimeProviderReceipt,
    scoring_observation_bytes: bytes | None = None,
    expected_outer_image_digest: str | None = None,
) -> tuple[str, ...]:
    """Return every provider-neutral cross-binding error in deterministic order."""

    observation_bytes = (
        scoring_observation_bytes
        if scoring_observation_bytes is not None
        else encode_scoring_observation(scoring_observation)
    )
    errors: list[str] = []
    if receipt.artifact_identity != artifact_identity:
        errors.append("receipt artifact_identity does not match the bound artifact")
    if receipt.scoring_observation_sha256 != _sha256(observation_bytes):
        errors.append(
            "receipt scoring_observation_sha256 does not match observation bytes"
        )
    provider_names = {
        receipt.plugin.name,
        receipt.capabilities.provider_name,
        scoring_observation.provider_name,
    }
    if len(provider_names) != 1:
        errors.append("receipt and observation provider names do not agree")
    if scoring_observation.artifact_identity_sha256 != artifact_identity_sha256(
        artifact_identity
    ):
        errors.append(
            "observation artifact_identity_sha256 does not match bound artifact"
        )
    if artifact_identity.artifact_format not in receipt.capabilities.artifact_formats:
        errors.append("bound artifact format is not declared by provider capabilities")
    if expected_outer_image_digest is not None:
        if _IMAGE_DIGEST.fullmatch(expected_outer_image_digest) is None:
            errors.append("expected outer image digest is not canonical")
        elif receipt.outer_image_digest != expected_outer_image_digest:
            errors.append(
                "receipt outer_image_digest does not match expected runtime image"
            )
    return tuple(errors)


def _require_cross_binding(
    *,
    artifact_identity: ModelArtifactIdentity,
    scoring_observation: ScoringObservation,
    receipt: RuntimeProviderReceipt,
    scoring_observation_bytes: bytes,
    expected_outer_image_digest: str | None,
) -> None:
    errors = runtime_provider_evidence_errors(
        artifact_identity=artifact_identity,
        scoring_observation=scoring_observation,
        receipt=receipt,
        scoring_observation_bytes=scoring_observation_bytes,
        expected_outer_image_digest=expected_outer_image_digest,
    )
    if errors:
        raise RuntimeProviderEvidenceError("; ".join(errors))


def _checked_sibling_paths(
    paths: RuntimeProviderEvidencePaths,
    *,
    require_existing: bool,
) -> tuple[Path, Path, Path]:
    candidates = (
        Path(paths.artifact_identity),
        Path(paths.scoring_observation),
        Path(paths.receipt),
    )
    expected_names = (
        ARTIFACT_IDENTITY_FILENAME,
        SCORING_OBSERVATION_FILENAME,
        PROVIDER_RECEIPT_FILENAME,
    )
    if tuple(path.name for path in candidates) != expected_names:
        raise RuntimeProviderEvidenceError(
            "runtime provider evidence paths must use the canonical filenames"
        )
    absolute = tuple(Path(os.path.abspath(path)) for path in candidates)
    if len(set(absolute)) != len(absolute):
        raise RuntimeProviderEvidenceError(
            "runtime provider evidence paths must be distinct"
        )
    if len({path.parent for path in absolute}) != 1:
        raise RuntimeProviderEvidenceError(
            "runtime provider evidence files must be siblings"
        )
    if require_existing:
        identities: list[tuple[int, int]] = []
        for path in absolute:
            try:
                file_stat = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeProviderEvidenceError(
                    f"unable to stat runtime provider evidence: {path.name}"
                ) from exc
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeProviderEvidenceError(
                    f"runtime provider evidence must be a regular file: {path.name}"
                )
            identities.append((file_stat.st_dev, file_stat.st_ino))
        if len(set(identities)) != len(identities):
            raise RuntimeProviderEvidenceError(
                "runtime provider evidence files must not alias the same file"
            )
    return cast(tuple[Path, Path, Path], absolute)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish one new sidecar without ever replacing an existing directory entry."""

    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        temporary = None
    except OSError as exc:
        raise RuntimeProviderEvidenceError(
            f"could not atomically write runtime provider evidence: {path.name}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_runtime_provider_evidence(
    directory: str | Path,
    *,
    artifact_identity: ModelArtifactIdentity,
    scoring_observation: ScoringObservation,
    receipt: RuntimeProviderReceipt,
    expected_outer_image_digest: str | None = None,
) -> PersistedRuntimeProviderEvidence:
    """Write a no-clobber canonical sidecar set and strictly reload it.

    Each file publication is atomic. Callers that require all-or-nothing directory
    publication must write into a private temporary directory and rename that
    directory only after this function returns successfully.
    """

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeProviderEvidenceError(
            "runtime provider evidence directory must be a real directory"
        )
    paths = RuntimeProviderEvidencePaths.in_directory(root)
    candidates = _checked_sibling_paths(paths, require_existing=False)
    existing = [path.name for path in candidates if path.exists() or path.is_symlink()]
    if existing:
        raise RuntimeProviderEvidenceError(
            "runtime provider evidence files already exist: " + ", ".join(existing)
        )

    artifact_bytes = encode_artifact_identity(artifact_identity)
    observation_bytes = encode_scoring_observation(scoring_observation)
    receipt_bytes = encode_runtime_provider_receipt(receipt)
    _require_cross_binding(
        artifact_identity=artifact_identity,
        scoring_observation=scoring_observation,
        receipt=receipt,
        scoring_observation_bytes=observation_bytes,
        expected_outer_image_digest=expected_outer_image_digest,
    )

    committed: list[Path] = []
    try:
        for path, payload in zip(
            candidates,
            (artifact_bytes, observation_bytes, receipt_bytes),
            strict=True,
        ):
            _atomic_write_bytes(path, payload)
            committed.append(path)
    except RuntimeProviderEvidenceError:
        for path in reversed(committed):
            path.unlink(missing_ok=True)
        raise
    return load_runtime_provider_evidence(
        paths,
        expected_outer_image_digest=expected_outer_image_digest,
    )


def load_runtime_provider_evidence(
    paths: RuntimeProviderEvidencePaths,
    *,
    expected_outer_image_digest: str | None = None,
) -> PersistedRuntimeProviderEvidence:
    """Strictly reload and type-check three sibling provider evidence files."""

    artifact_path, observation_path, receipt_path = _checked_sibling_paths(
        paths, require_existing=True
    )
    try:
        artifact_bytes = read_regular_file_bytes(
            artifact_path,
            label="model artifact identity",
            max_bytes=MAX_RUNTIME_PROVIDER_SIDECAR_BYTES,
        )
        observation_bytes = read_regular_file_bytes(
            observation_path,
            label="runtime scoring observation",
            max_bytes=MAX_RUNTIME_PROVIDER_SIDECAR_BYTES,
        )
        receipt_bytes = read_regular_file_bytes(
            receipt_path,
            label="runtime provider receipt",
            max_bytes=MAX_RUNTIME_PROVIDER_SIDECAR_BYTES,
        )
    except StrictJsonError as exc:
        raise RuntimeProviderEvidenceError(str(exc)) from exc

    artifact_identity = decode_artifact_identity(artifact_bytes)
    scoring_observation = decode_scoring_observation(observation_bytes)
    receipt = decode_runtime_provider_receipt(receipt_bytes)
    _require_cross_binding(
        artifact_identity=artifact_identity,
        scoring_observation=scoring_observation,
        receipt=receipt,
        scoring_observation_bytes=observation_bytes,
        expected_outer_image_digest=expected_outer_image_digest,
    )
    return PersistedRuntimeProviderEvidence(
        paths=RuntimeProviderEvidencePaths(
            artifact_identity=artifact_path,
            scoring_observation=observation_path,
            receipt=receipt_path,
        ),
        artifact_identity=artifact_identity,
        scoring_observation=scoring_observation,
        receipt=receipt,
        artifact_identity_bytes=artifact_bytes,
        scoring_observation_bytes=observation_bytes,
        receipt_bytes=receipt_bytes,
        artifact_identity_sha256=_sha256(artifact_bytes),
        scoring_observation_sha256=_sha256(observation_bytes),
        receipt_sha256=_sha256(receipt_bytes),
    )


__all__ = [
    "ARTIFACT_IDENTITY_FILENAME",
    "MAX_RUNTIME_PROVIDER_SIDECAR_BYTES",
    "PROVIDER_RECEIPT_FILENAME",
    "PersistedRuntimeProviderEvidence",
    "RuntimeProviderEvidenceError",
    "RuntimeProviderEvidencePaths",
    "SCORING_OBSERVATION_FILENAME",
    "decode_artifact_identity",
    "decode_runtime_provider_capabilities",
    "decode_runtime_provider_receipt",
    "decode_scoring_observation",
    "encode_artifact_identity",
    "encode_runtime_provider_capabilities",
    "encode_runtime_provider_receipt",
    "encode_scoring_observation",
    "load_runtime_provider_evidence",
    "runtime_provider_evidence_errors",
    "write_runtime_provider_evidence",
]
