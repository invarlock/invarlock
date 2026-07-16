"""Torch-free value contracts for runtime-provider execution and evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAliasType

RUNTIME_PROVIDER_ABI_VERSION = "1"
RUNTIME_PROVIDER_CAPABILITIES_FORMAT = "runtime-provider-capabilities-v1"
MODEL_ARTIFACT_IDENTITY_FORMAT = "invarlock/model-artifact-identity-v1"
RUNTIME_PROVIDER_RECEIPT_FORMAT = "invarlock/runtime-provider-receipt-v1"
RUNTIME_SCORING_OBSERVATION_FORMAT = "invarlock/runtime-scoring-observation-v1"

ArtifactFormat = TypeAliasType(  # noqa: UP040
    "ArtifactFormat", Literal["hf_snapshot", "gguf", "tensorrt_llm_engine"]
)
RuntimeTask = TypeAliasType("RuntimeTask", str)  # noqa: UP040
EvaluationInputPartKind = TypeAliasType(  # noqa: UP040
    "EvaluationInputPartKind", Literal["text", "content"]
)
RuntimeMetric = TypeAliasType(  # noqa: UP040
    "RuntimeMetric",
    Literal[
        "exact_match",
        "normalized_nll_per_utf8_byte",
    ],
)
RuntimeExecutionMode = TypeAliasType(  # noqa: UP040
    "RuntimeExecutionMode", Literal["in_process", "local_process", "container"]
)
ScoringStatus = TypeAliasType(  # noqa: UP040
    "ScoringStatus", Literal["ok", "error"]
)
JSONScalar = TypeAliasType(  # noqa: UP040
    "JSONScalar", str | int | float | bool | None
)

_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SETTING_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_COMPUTE_CAPABILITY = re.compile(r"^(0|[1-9][0-9]?)\.(0|[1-9][0-9]?)$")
_IMMUTABLE_REMOTE_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_RESOURCE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TASK_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MEDIA_TYPE = re.compile(
    r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,126}/[a-z0-9][a-z0-9!#$&^_.+-]{0,126}$"
)

_ARTIFACT_FORMATS = frozenset({"hf_snapshot", "gguf", "tensorrt_llm_engine"})
STANDARD_RUNTIME_TASKS = frozenset(
    {
        "text_causal",
        "text_seq2seq",
        "masked_language",
        "vision_text_generation",
    }
)
_METRICS = frozenset({"exact_match", "normalized_nll_per_utf8_byte"})
_EXECUTION_MODES = frozenset({"in_process", "local_process", "container"})


def _require_nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _require_provider_name(value: object, *, field_name: str) -> str:
    text = _require_nonempty_string(value, field_name=field_name)
    if _PROVIDER_NAME.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a canonical provider name")
    return text


def require_runtime_task(value: object, *, field_name: str = "task") -> RuntimeTask:
    """Return one canonical task identifier or fail closed.

    The ABI reserves the identifiers in ``STANDARD_RUNTIME_TASKS`` but
    intentionally accepts any canonical snake-case identifier. Optional add-ins
    can therefore declare later modalities without an ABI or schema revision.
    """

    text = _require_nonempty_string(value, field_name=field_name)
    if _TASK_NAME.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a canonical task identifier")
    return text


def _require_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")
    return value


def _require_optional_sha256(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, field_name=field_name)


def _require_image_digest(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _IMAGE_DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a sha256 image digest")
    return value


def _require_safe_logical_name(value: object, *, field_name: str) -> str:
    text = _require_nonempty_string(value, field_name=field_name)
    normalized = text.replace("\\", "/")
    if normalized.startswith("/") or any(
        part in {"", ".", ".."} for part in normalized.split("/")
    ):
        raise ValueError(f"{field_name} must not be an absolute or traversal path")
    if ":/" in normalized or (len(normalized) >= 2 and normalized[1] == ":"):
        raise ValueError(f"{field_name} must not be an absolute or traversal path")
    return text


def _require_safe_basename(value: object, *, field_name: str) -> str:
    text = _require_safe_logical_name(value, field_name=field_name)
    if "/" in text or "\\" in text:
        raise ValueError(f"{field_name} must not be an absolute or traversal path")
    return text


def _require_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_unique_allowed(
    values: tuple[str, ...],
    *,
    field_name: str,
    allowed: frozenset[str],
    item_label: str,
    allow_empty: bool = False,
) -> None:
    if not isinstance(values, tuple) or (not values and not allow_empty):
        raise ValueError(f"{field_name} must be a non-empty tuple")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicate values")
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"unsupported {item_label}: {invalid[0]}")


@dataclass(frozen=True)
class RuntimeProviderCapabilities:
    """Closed execution capabilities declared by one provider."""

    provider_name: str
    artifact_formats: tuple[ArtifactFormat, ...]
    tasks: tuple[RuntimeTask, ...]
    metrics: tuple[RuntimeMetric, ...]
    execution_modes: tuple[RuntimeExecutionMode, ...]
    required_extra: str | None
    required_image: str | None
    format_version: str = field(
        default=RUNTIME_PROVIDER_CAPABILITIES_FORMAT, init=False
    )
    provider_abi: str = field(default=RUNTIME_PROVIDER_ABI_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_provider_name(self.provider_name, field_name="provider_name")
        _require_unique_allowed(
            self.artifact_formats,
            field_name="artifact_formats",
            allowed=_ARTIFACT_FORMATS,
            item_label="artifact format",
        )
        _require_unique_allowed(
            self.tasks,
            field_name="tasks",
            allowed=frozenset(self.tasks),
            item_label="runtime task",
        )
        for task in self.tasks:
            require_runtime_task(task, field_name="tasks entry")
        _require_unique_allowed(
            self.metrics,
            field_name="metrics",
            allowed=_METRICS,
            item_label="runtime metric",
        )
        _require_unique_allowed(
            self.execution_modes,
            field_name="execution_modes",
            allowed=_EXECUTION_MODES,
            item_label="execution mode",
        )
        if self.required_extra is not None:
            _require_provider_name(self.required_extra, field_name="required_extra")
        if self.required_image is not None:
            _require_nonempty_string(self.required_image, field_name="required_image")
            if any(character.isspace() for character in self.required_image):
                raise ValueError("required_image must not contain whitespace")


@dataclass(frozen=True)
class HFSnapshotArtifactIdentity:
    """Immutable Hugging Face snapshot or secure checkpoint-tree identity."""

    model_id: str
    immutable_revision: str | None
    checkpoint_tree_sha256: str | None
    tokenizer_metadata_sha256: str
    format_version: str = field(default=MODEL_ARTIFACT_IDENTITY_FORMAT, init=False)
    artifact_format: ArtifactFormat = field(default="hf_snapshot", init=False)

    def __post_init__(self) -> None:
        _require_safe_logical_name(self.model_id, field_name="model_id")
        if self.immutable_revision is not None:
            if _IMMUTABLE_REMOTE_REVISION.fullmatch(self.immutable_revision) is None:
                raise ValueError(
                    "immutable_revision must be a 40-64 character lowercase "
                    "hexadecimal revision"
                )
        _require_optional_sha256(
            self.checkpoint_tree_sha256, field_name="checkpoint_tree_sha256"
        )
        _require_sha256(
            self.tokenizer_metadata_sha256, field_name="tokenizer_metadata_sha256"
        )
        if self.immutable_revision is None and self.checkpoint_tree_sha256 is None:
            raise ValueError(
                "at least one of immutable_revision or checkpoint_tree_sha256 is required"
            )


@dataclass(frozen=True)
class GGUFArtifactIdentity:
    """Content and metadata identity for one immutable GGUF regular file."""

    artifact_name: str
    sha256: str
    byte_length: int
    gguf_metadata_sha256: str
    tensor_inventory_sha256: str
    tokenizer_metadata_sha256: str
    format_version: str = field(default=MODEL_ARTIFACT_IDENTITY_FORMAT, init=False)
    artifact_format: ArtifactFormat = field(default="gguf", init=False)

    def __post_init__(self) -> None:
        _require_safe_basename(self.artifact_name, field_name="artifact_name")
        _require_sha256(self.sha256, field_name="sha256")
        _require_positive_int(self.byte_length, field_name="byte_length")
        _require_sha256(self.gguf_metadata_sha256, field_name="gguf_metadata_sha256")
        _require_sha256(
            self.tensor_inventory_sha256, field_name="tensor_inventory_sha256"
        )
        _require_sha256(
            self.tokenizer_metadata_sha256, field_name="tokenizer_metadata_sha256"
        )


@dataclass(frozen=True)
class TensorRTLLMArtifactIdentity:
    """Content identity for an engine bundle and its external tokenizer."""

    bundle_name: str
    engine_bundle_tree_sha256: str
    file_inventory_sha256: str
    builder_config_sha256: str
    tokenizer_metadata_sha256: str
    engine_metadata_sha256: str
    target_compute_capability: str
    format_version: str = field(default=MODEL_ARTIFACT_IDENTITY_FORMAT, init=False)
    artifact_format: ArtifactFormat = field(default="tensorrt_llm_engine", init=False)

    def __post_init__(self) -> None:
        _require_safe_basename(self.bundle_name, field_name="bundle_name")
        _require_sha256(
            self.engine_bundle_tree_sha256, field_name="engine_bundle_tree_sha256"
        )
        _require_sha256(self.file_inventory_sha256, field_name="file_inventory_sha256")
        _require_sha256(self.builder_config_sha256, field_name="builder_config_sha256")
        _require_sha256(
            self.tokenizer_metadata_sha256,
            field_name="tokenizer_metadata_sha256",
        )
        _require_sha256(
            self.engine_metadata_sha256, field_name="engine_metadata_sha256"
        )
        if _COMPUTE_CAPABILITY.fullmatch(self.target_compute_capability) is None:
            raise ValueError("target_compute_capability must use major.minor notation")


ModelArtifactIdentity = TypeAliasType(  # noqa: UP040
    "ModelArtifactIdentity",
    HFSnapshotArtifactIdentity | GGUFArtifactIdentity | TensorRTLLMArtifactIdentity,
)


def canonical_artifact_identity_json(identity: ModelArtifactIdentity) -> bytes:
    """Serialize an artifact identity using the sole receipt/hash convention."""

    if not isinstance(
        identity,
        (
            HFSnapshotArtifactIdentity,
            GGUFArtifactIdentity,
            TensorRTLLMArtifactIdentity,
        ),
    ):
        raise TypeError("identity must be a supported model artifact identity")
    return json.dumps(
        asdict(identity),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def artifact_identity_sha256(identity: ModelArtifactIdentity) -> str:
    """Return the bare lowercase SHA-256 used by scoring observations."""

    return hashlib.sha256(canonical_artifact_identity_json(identity)).hexdigest()


@dataclass(frozen=True)
class ModelRuntimeSpec:
    """Provider selection plus local runtime inputs; this object is not a receipt."""

    provider_name: str
    model_id: str
    settings: Mapping[str, JSONScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_provider_name(self.provider_name, field_name="provider_name")
        _require_nonempty_string(self.model_id, field_name="model_id")
        if not isinstance(self.settings, Mapping):
            raise ValueError("settings must be a mapping of JSON scalar values")
        copied: dict[str, JSONScalar] = {}
        for key, value in self.settings.items():
            if not isinstance(key, str) or _SETTING_NAME.fullmatch(key) is None:
                raise ValueError("settings keys must be canonical names")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError("settings values must be JSON scalar values")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("settings values must be finite JSON scalar values")
            copied[key] = value
        object.__setattr__(self, "settings", MappingProxyType(copied))


@dataclass(frozen=True)
class EvaluationInputPart:
    """One ordered, portable input component for a future-capable provider.

    Text is carried inline and authenticated directly. Binary or media content is
    represented only by a canonical caller-owned content identifier, media type,
    byte length, and digest. A schedule never carries a host path or URI.
    """

    kind: EvaluationInputPartKind
    role: str
    sha256: str
    text: str | None = None
    content_id: str | None = None
    media_type: str | None = None
    byte_length: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"text", "content"}:
            raise ValueError("input part kind must be text or content")
        _require_provider_name(self.role, field_name="input part role")
        _require_sha256(self.sha256, field_name="input part sha256")
        if self.kind == "text":
            if not isinstance(self.text, str) or not self.text.strip():
                raise ValueError("text input part must contain non-empty text")
            if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.sha256:
                raise ValueError("text input part sha256 does not match text")
            if any(
                value is not None
                for value in (self.content_id, self.media_type, self.byte_length)
            ):
                raise ValueError("text input part must not contain content fields")
            return

        if self.text is not None:
            raise ValueError("content input part must not contain inline text")
        _require_provider_name(
            self.content_id, field_name="content input part content_id"
        )
        media_type = _require_nonempty_string(
            self.media_type, field_name="content input part media_type"
        )
        if _MEDIA_TYPE.fullmatch(media_type) is None:
            raise ValueError("content input part media_type must be canonical")
        _require_positive_int(
            self.byte_length, field_name="content input part byte_length"
        )

    def to_payload(self) -> dict[str, object]:
        """Return the closed public representation for canonical serialization."""

        if self.kind == "text":
            return {
                "kind": self.kind,
                "role": self.role,
                "text": self.text,
                "sha256": self.sha256,
            }
        return {
            "kind": self.kind,
            "role": self.role,
            "content_id": self.content_id,
            "media_type": self.media_type,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }


def canonical_evaluation_input_parts_json(
    parts: tuple[EvaluationInputPart, ...],
) -> bytes:
    """Serialize ordered structured inputs using the sole schedule convention."""

    if not isinstance(parts, tuple) or not parts:
        raise ValueError("input_parts must be a non-empty tuple")
    return json.dumps(
        [part.to_payload() for part in parts],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def evaluation_input_parts_sha256(parts: tuple[EvaluationInputPart, ...]) -> str:
    """Authenticate the exact ordered structured input material."""

    return hashlib.sha256(canonical_evaluation_input_parts_json(parts)).hexdigest()


@dataclass(frozen=True)
class EvaluationRecord:
    """One canonically identified evaluation input."""

    record_id: str
    input_text: str
    input_sha256: str
    expected_output: str | None = None
    input_parts: tuple[EvaluationInputPart, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.record_id, field_name="record_id")
        if not isinstance(self.input_text, str):
            raise ValueError("input_text must be a string")
        _require_sha256(self.input_sha256, field_name="input_sha256")
        if self.expected_output is not None and not isinstance(
            self.expected_output, str
        ):
            raise ValueError("expected_output must be a string or null")
        if not isinstance(self.input_parts, tuple):
            raise ValueError("input_parts must be a tuple")
        if not self.input_parts:
            return
        roles = [part.role for part in self.input_parts]
        if len(roles) != len(set(roles)):
            raise ValueError("input part roles must be unique within a record")
        text_parts = [part for part in self.input_parts if part.kind == "text"]
        if not text_parts:
            raise ValueError("structured input requires at least one text part")
        if text_parts[0].text != self.input_text:
            raise ValueError("input_text must match the first structured text part")
        if evaluation_input_parts_sha256(self.input_parts) != self.input_sha256:
            raise ValueError("input_sha256 does not match ordered input_parts")


@dataclass(frozen=True)
class EvaluationBatch:
    """A schedule-bound, uniquely paired evaluation batch."""

    schedule_sha256: str
    records: tuple[EvaluationRecord, ...]
    metric: RuntimeMetric = "exact_match"
    task: RuntimeTask = "text_causal"

    def __post_init__(self) -> None:
        _require_sha256(self.schedule_sha256, field_name="schedule_sha256")
        if self.metric not in _METRICS:
            raise ValueError(f"unsupported runtime metric: {self.metric}")
        require_runtime_task(self.task, field_name="task")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("records must be a non-empty tuple")
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("record IDs must be unique within a batch")


@dataclass(frozen=True)
class RuntimeScoringRecord:
    """Backend-measured facts for one input; aggregates are verifier-owned."""

    record_id: str
    input_sha256: str
    status: ScoringStatus
    output_text: str | None = None
    output_sha256: str | None = None
    logprob_sum: float | None = None
    token_count: int | None = None
    utf8_byte_count: int | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.record_id, field_name="record_id")
        _require_sha256(self.input_sha256, field_name="input_sha256")
        if self.status not in {"ok", "error"}:
            raise ValueError("status must be ok or error")
        if self.output_text is not None and not isinstance(self.output_text, str):
            raise ValueError("output_text must be a string or null")
        _require_optional_sha256(self.output_sha256, field_name="output_sha256")
        if (self.output_text is None) != (self.output_sha256 is None):
            raise ValueError("output_text and output_sha256 must be present together")
        if self.logprob_sum is not None and (
            isinstance(self.logprob_sum, bool)
            or not isinstance(self.logprob_sum, (int, float))
            or not math.isfinite(float(self.logprob_sum))
        ):
            raise ValueError("logprob_sum must be finite")
        if self.token_count is not None:
            _require_nonnegative_int(self.token_count, field_name="token_count")
        if self.utf8_byte_count is not None:
            _require_nonnegative_int(self.utf8_byte_count, field_name="utf8_byte_count")
        if self.logprob_sum is not None and (
            self.token_count is None
            or self.token_count <= 0
            or self.utf8_byte_count is None
            or self.utf8_byte_count <= 0
        ):
            raise ValueError(
                "logprob_sum requires positive token_count and utf8_byte_count"
            )
        if self.status == "error":
            if self.error_code is None:
                raise ValueError("error_code is required when status is error")
            _require_provider_name(self.error_code, field_name="error_code")
            if any(
                value is not None
                for value in (
                    self.output_text,
                    self.output_sha256,
                    self.logprob_sum,
                    self.token_count,
                    self.utf8_byte_count,
                )
            ):
                raise ValueError("error records must not contain measured output facts")
        elif self.error_code is not None:
            raise ValueError("error_code must be null when status is ok")
        elif self.output_text is None and self.logprob_sum is None:
            raise ValueError("ok records require output or logprob facts")


@dataclass(frozen=True)
class ScoringObservation:
    """Provider output bound to one artifact and one paired schedule."""

    provider_name: str
    artifact_identity_sha256: str
    schedule_sha256: str
    records: tuple[RuntimeScoringRecord, ...]
    aggregate_source_sha256: str
    format_version: str = field(default=RUNTIME_SCORING_OBSERVATION_FORMAT, init=False)

    def __post_init__(self) -> None:
        _require_provider_name(self.provider_name, field_name="provider_name")
        _require_sha256(
            self.artifact_identity_sha256, field_name="artifact_identity_sha256"
        )
        _require_sha256(self.schedule_sha256, field_name="schedule_sha256")
        _require_sha256(
            self.aggregate_source_sha256, field_name="aggregate_source_sha256"
        )
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("records must be a non-empty tuple")
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("record IDs must be unique within an observation")


@dataclass(frozen=True)
class RuntimeExecutionSettings:
    """Shared deterministic settings captured in provider receipts."""

    seed: int
    context_length: int
    batch_size: int
    max_output_tokens: int
    timeout_seconds: int
    allow_network: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        _require_positive_int(self.context_length, field_name="context_length")
        _require_positive_int(self.batch_size, field_name="batch_size")
        _require_positive_int(self.max_output_tokens, field_name="max_output_tokens")
        _require_positive_int(self.timeout_seconds, field_name="timeout_seconds")
        if not isinstance(self.allow_network, bool):
            raise ValueError("allow_network must be boolean")


def runtime_execution_settings_from_mapping(
    settings: Mapping[str, JSONScalar],
    *,
    allow_network: bool,
) -> RuntimeExecutionSettings:
    """Build the canonical portable execution settings from provider settings."""

    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping")
    return RuntimeExecutionSettings(
        seed=_require_nonnegative_int(settings.get("seed"), field_name="seed"),
        context_length=_require_positive_int(
            settings.get("context_length"), field_name="context_length"
        ),
        batch_size=_require_positive_int(
            settings.get("batch_size"), field_name="batch_size"
        ),
        max_output_tokens=_require_positive_int(
            settings.get("max_output_tokens"), field_name="max_output_tokens"
        ),
        timeout_seconds=_require_positive_int(
            settings.get("timeout_seconds"), field_name="timeout_seconds"
        ),
        allow_network=allow_network,
    )


RuntimeScorer = TypeAliasType(  # noqa: UP040
    "RuntimeScorer",
    Callable[[EvaluationBatch, RuntimeExecutionSettings], ScoringObservation],
)


def _validate_resource_path(root: Path, relative_path: str, *, label: str) -> Path:
    """Revalidate one root-confined resource without following symbolic links."""

    if not isinstance(relative_path, str) or "\\" in relative_path:
        raise ValueError(f"{label} must be a portable relative path")
    _require_safe_logical_name(relative_path, field_name=label)
    parts = tuple(relative_path.split("/"))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    leaf_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[int] = []
    try:
        current = os.open(root, directory_flags)
        descriptors.append(current)
        for component in parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
        leaf = os.open(parts[-1], leaf_flags, dir_fd=current)
        descriptors.append(leaf)
        mode = os.fstat(leaf).st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"{label} must name a regular file or directory")
    except OSError as exc:
        raise ValueError(
            f"{label} must exist beneath root without symbolic links"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return root.joinpath(*parts)


@dataclass(frozen=True)
class RuntimeArtifactResources:
    """Closed, offline resources supplied by the caller to one provider.

    Paths are portable names beneath an absolute caller-owned root. Construction
    and every lookup authenticate the path component-by-component without
    following symbolic links. The object deliberately has no network, remote-code,
    plugin, or arbitrary host-capability switches: evaluation cannot
    acquire those permissions from request data.
    """

    root: Path = field(repr=False)
    primary_artifact: str = field(repr=False)
    support_resources: Mapping[str, str] = field(repr=False, compare=False)
    device_kind: str
    container_image_digest: str

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute():
            raise ValueError("runtime artifact resource root must be absolute")
        root_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(root, root_flags)
        except OSError as exc:
            raise ValueError(
                "runtime artifact resource root must be a non-symlink directory"
            ) from exc
        else:
            os.close(descriptor)
        if self.device_kind not in {"cpu", "cuda"}:
            raise ValueError("device_kind must be cpu or cuda")
        if (
            _require_image_digest(
                self.container_image_digest,
                field_name="container_image_digest",
            )
            is None
        ):
            raise ValueError("container_image_digest is required")

        support = dict(self.support_resources)
        for name, relative_path in support.items():
            if not isinstance(name, str) or _RESOURCE_NAME.fullmatch(name) is None:
                raise ValueError("support resource names must be canonical identifiers")
            _validate_resource_path(
                root,
                relative_path,
                label=f"support resource {name!r}",
            )
        _validate_resource_path(root, self.primary_artifact, label="primary_artifact")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "support_resources", MappingProxyType(support))

    def primary_path(self) -> Path:
        """Return the primary resource after repeating no-follow validation."""

        return _validate_resource_path(
            self.root, self.primary_artifact, label="primary_artifact"
        )

    def support_path(self, name: str) -> Path:
        """Return one named support resource after repeating validation."""

        try:
            relative_path = self.support_resources[name]
        except KeyError as exc:
            raise ValueError(f"missing required support resource {name!r}") from exc
        return _validate_resource_path(
            self.root,
            relative_path,
            label=f"support resource {name!r}",
        )

    def require_support_names(self, expected: frozenset[str]) -> None:
        """Reject missing and undeclared provider support resources."""

        observed = set(self.support_resources)
        missing = expected - observed
        if missing:
            raise ValueError(
                f"missing required support resource {sorted(missing)[0]!r}"
            )
        unexpected = observed - expected
        if unexpected:
            raise ValueError(f"unexpected support resource {sorted(unexpected)[0]!r}")


@dataclass(frozen=True)
class RuntimeExecutionContext:
    """Ephemeral execution bindings owned by one runtime provider.

    ``provider_state`` is opaque outside the provider implementation. The public
    ABI exposes scoring and receipts, not mutable backend or model objects.
    """

    strict: bool
    allow_network: bool
    container_image_digest: str | None
    device_kind: str
    artifact_identity_sha256: str | None = None
    provider_state: object | None = field(default=None, repr=False, compare=False)
    scorer: RuntimeScorer | None = field(default=None, repr=False, compare=False)
    close_callback: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.strict, bool):
            raise ValueError("strict must be boolean")
        if not isinstance(self.allow_network, bool):
            raise ValueError("allow_network must be boolean")
        _require_image_digest(
            self.container_image_digest, field_name="container_image_digest"
        )
        _require_provider_name(self.device_kind, field_name="device_kind")
        _require_optional_sha256(
            self.artifact_identity_sha256, field_name="artifact_identity_sha256"
        )
        if self.scorer is not None and not callable(self.scorer):
            raise ValueError("scorer must be callable")
        if self.close_callback is not None and not callable(self.close_callback):
            raise ValueError("close_callback must be callable")


@dataclass(frozen=True)
class RuntimeProviderPluginIdentity:
    name: str
    distribution: str
    distribution_version: str
    provider_abi: str = field(default=RUNTIME_PROVIDER_ABI_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_provider_name(self.name, field_name="name")
        _require_nonempty_string(self.distribution, field_name="distribution")
        _require_nonempty_string(
            self.distribution_version, field_name="distribution_version"
        )


@dataclass(frozen=True)
class RuntimeBackendIdentity:
    name: str
    version: str
    source_sha256: str | None
    binary_sha256: str | None
    build_sha256: str | None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, field_name="name")
        _require_nonempty_string(self.version, field_name="version")
        for field_name in ("source_sha256", "binary_sha256", "build_sha256"):
            _require_optional_sha256(getattr(self, field_name), field_name=field_name)
        if all(
            value is None
            for value in (self.source_sha256, self.binary_sha256, self.build_sha256)
        ):
            raise ValueError("at least one backend digest is required")


@dataclass(frozen=True)
class RuntimeDeviceFacts:
    device_kind: str
    device_name: str
    compute_capability: str | None = None
    driver_version: str | None = None
    cuda_runtime_version: str | None = None

    def __post_init__(self) -> None:
        _require_provider_name(self.device_kind, field_name="device_kind")
        _require_nonempty_string(self.device_name, field_name="device_name")
        if (
            self.compute_capability is not None
            and _COMPUTE_CAPABILITY.fullmatch(self.compute_capability) is None
        ):
            raise ValueError("compute_capability must use major.minor notation")
        if self.driver_version is not None:
            _require_nonempty_string(self.driver_version, field_name="driver_version")
        if self.cuda_runtime_version is not None:
            _require_nonempty_string(
                self.cuda_runtime_version, field_name="cuda_runtime_version"
            )


@dataclass(frozen=True)
class RuntimeProviderReceipt:
    """Canonical provider provenance bound to artifact, settings, and observations."""

    plugin: RuntimeProviderPluginIdentity
    backend: RuntimeBackendIdentity
    capabilities: RuntimeProviderCapabilities
    artifact_identity: ModelArtifactIdentity
    execution_settings: RuntimeExecutionSettings
    device: RuntimeDeviceFacts
    outer_image_digest: str | None
    scoring_observation_sha256: str
    format_version: str = field(default=RUNTIME_PROVIDER_RECEIPT_FORMAT, init=False)

    def __post_init__(self) -> None:
        if self.plugin.name != self.capabilities.provider_name:
            raise ValueError("plugin and capabilities provider names must match")
        _require_image_digest(self.outer_image_digest, field_name="outer_image_digest")
        _require_sha256(
            self.scoring_observation_sha256,
            field_name="scoring_observation_sha256",
        )


__all__ = [
    "ArtifactFormat",
    "artifact_identity_sha256",
    "STANDARD_RUNTIME_TASKS",
    "canonical_artifact_identity_json",
    "canonical_evaluation_input_parts_json",
    "EvaluationBatch",
    "EvaluationInputPart",
    "EvaluationInputPartKind",
    "EvaluationRecord",
    "evaluation_input_parts_sha256",
    "GGUFArtifactIdentity",
    "HFSnapshotArtifactIdentity",
    "JSONScalar",
    "MODEL_ARTIFACT_IDENTITY_FORMAT",
    "ModelArtifactIdentity",
    "ModelRuntimeSpec",
    "RUNTIME_PROVIDER_ABI_VERSION",
    "RUNTIME_PROVIDER_CAPABILITIES_FORMAT",
    "RUNTIME_PROVIDER_RECEIPT_FORMAT",
    "RUNTIME_SCORING_OBSERVATION_FORMAT",
    "RuntimeBackendIdentity",
    "RuntimeArtifactResources",
    "RuntimeDeviceFacts",
    "RuntimeExecutionContext",
    "RuntimeExecutionMode",
    "RuntimeExecutionSettings",
    "runtime_execution_settings_from_mapping",
    "RuntimeMetric",
    "RuntimeProviderCapabilities",
    "RuntimeProviderPluginIdentity",
    "RuntimeProviderReceipt",
    "RuntimeScorer",
    "RuntimeScoringRecord",
    "RuntimeTask",
    "require_runtime_task",
    "ScoringObservation",
    "ScoringStatus",
    "TensorRTLLMArtifactIdentity",
]
