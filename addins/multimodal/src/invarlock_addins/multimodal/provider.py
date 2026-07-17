"""Authenticated offline Hugging Face image-and-text generation provider."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import platform
import re
import stat
import time
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

from invarlock_addins.multimodal import __version__ as ADDIN_VERSION

from invarlock.core.checkpoint_identity import (
    CheckpointIdentityError,
    checkpoint_tree_sha256,
)
from invarlock.core.runtime_provider import (
    INVARLOCK_RUNTIME_PROVIDER_ABI,
    EvaluationBatch,
    EvaluationInputPart,
    EvaluationRecord,
    HFSnapshotArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeBackendIdentity,
    RuntimeBehavioralSchedule,
    RuntimeDeviceFacts,
    RuntimeExecutionContext,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    RuntimeProviderReceipt,
    RuntimeScoringRecord,
    ScoringObservation,
    artifact_identity_sha256,
    exact_match_output_text,
    runtime_execution_settings_from_mapping,
)
from invarlock.core.runtime_provider.behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.core.runtime_provider.types import JSONScalar
from invarlock.runtime_provider_evidence import (
    encode_scoring_observation,
)
from invarlock.runtime_providers.hf_transformers import (
    hf_tokenizer_contract_sha256,
    load_hf_model_with_strict_loading_info,
    require_loaded_hf_checkpoint_binding,
)
from invarlock.runtime_security_helpers import (
    network_allowed,
    remote_code_allowed,
    resolve_runtime_image,
    resolve_runtime_image_digest,
    strict_container_boundary_present,
    third_party_plugins_allowed,
)

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_CONTENT_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,255}$")
_ALLOWED_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_MEDIA_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_MAX_IMAGE_PIXELS = 50_000_000
_MAX_UNIQUE_IMAGES = 2_048
_MAX_TOTAL_IMAGE_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_IMAGE_PIXELS = 200_000_000
_ALLOWED_SETTINGS = frozenset(
    {
        "batch_size",
        "checkpoint_tree_sha256",
        "context_length",
        "immutable_revision",
        "max_output_tokens",
        "offline",
        "processor_metadata_sha256",
        "seed",
        "timeout_seconds",
        "tokenizer_metadata_sha256",
    }
)
_REQUIRED_SETTINGS = frozenset(
    {
        "batch_size",
        "checkpoint_tree_sha256",
        "context_length",
        "max_output_tokens",
        "offline",
        "processor_metadata_sha256",
        "seed",
        "timeout_seconds",
        "tokenizer_metadata_sha256",
    }
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_digest(settings: Mapping[str, JSONScalar], name: str) -> str:
    value = settings.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a sha256 digest")
    digest = value.removeprefix("sha256:")
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a sha256 digest")
    return digest


def _optional_text(settings: Mapping[str, JSONScalar], name: str) -> str | None:
    value = settings.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty trimmed text")
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


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise ValueError("processor contract contains a non-JSON value")


def processor_contract_sha256(processor: object) -> str:
    """Hash the live image processor, tokenizer, and chat-template contract."""

    tokenizer = getattr(processor, "tokenizer", None)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if tokenizer is None or not callable(get_vocab):
        raise ValueError("vision-text processor must expose a tokenizer vocabulary")
    vocab = get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise ValueError("vision-text tokenizer vocabulary is unavailable")
    normalized_vocab: dict[str, int] = {}
    for token, token_id in vocab.items():
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError("vision-text tokenizer vocabulary is invalid")
        normalized_vocab[str(token)] = token_id
    image_processor = getattr(processor, "image_processor", None)
    image_to_dict = getattr(image_processor, "to_dict", None)
    if not callable(image_to_dict):
        raise ValueError("vision-text image processor configuration is unavailable")
    image_config = image_to_dict()
    if not isinstance(image_config, Mapping) or not image_config:
        raise ValueError("vision-text image processor configuration is unavailable")
    chat_template = getattr(processor, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("vision-text chat template is unavailable")
    payload = {
        "format_version": "invarlock/hf-vision-text-processor-contract-v1",
        "processor_class": (
            processor.__class__.__module__ + "." + processor.__class__.__qualname__
        ),
        "image_processor": _json_safe(image_config),
        "tokenizer_class": (
            tokenizer.__class__.__module__ + "." + tokenizer.__class__.__qualname__
        ),
        "tokenizer_vocab": normalized_vocab,
        "special_tokens": _json_safe(getattr(tokenizer, "special_tokens_map", {})),
        "chat_template": chat_template,
    }
    return _sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_content_bytes(
    content_store: Path,
    *,
    content_id: str,
    expected_sha256: str,
    expected_byte_length: int,
) -> bytes:
    """Open one named content object without links and authenticate exact bytes."""

    if _CONTENT_ID.fullmatch(content_id) is None or content_id in {".", ".."}:
        raise ValueError("vision content_id must be a safe basename")
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("vision content digest must be lowercase sha256")
    if (
        isinstance(expected_byte_length, bool)
        or not isinstance(expected_byte_length, int)
        or expected_byte_length <= 0
        or expected_byte_length > _MAX_IMAGE_BYTES
    ):
        raise ValueError("vision content byte length is outside the supported range")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory = os.open(content_store, directory_flags)
    except OSError as exc:
        raise ValueError("vision content store is unavailable") from exc
    try:
        try:
            named = os.stat(content_id, dir_fd=directory, follow_symlinks=False)
            descriptor = os.open(content_id, file_flags, dir_fd=directory)
        except OSError as exc:
            raise ValueError("vision content object is unavailable") from exc
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or _stat_identity(named) != _stat_identity(opened)
                or opened.st_size != expected_byte_length
            ):
                raise ValueError("vision content object identity does not match")
            remaining = expected_byte_length
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    raise ValueError("vision content object changed while being read")
                chunks.append(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError("vision content object changed while being read")
            final_stat = os.fstat(descriptor)
            final_named = os.stat(content_id, dir_fd=directory, follow_symlinks=False)
            if (
                _stat_identity(final_stat) != _stat_identity(opened)
                or _stat_identity(final_named) != _stat_identity(opened)
                or digest.hexdigest() != expected_sha256
            ):
                raise ValueError("vision content object digest does not match")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _decode_image(payload: bytes, *, media_type: str) -> object:
    if media_type not in _ALLOWED_MEDIA_TYPES:
        raise ValueError("vision content media type is unsupported")
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise RuntimeError("Pillow is required for vision-text evaluation") from exc
    decompression_warning = getattr(image_module, "DecompressionBombWarning", Warning)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", decompression_warning)
            with image_module.open(io.BytesIO(payload)) as candidate:
                if candidate.format != _MEDIA_FORMATS[media_type]:
                    raise ValueError("vision content format does not match media_type")
                if int(getattr(candidate, "n_frames", 1)) != 1:
                    raise ValueError("animated vision content is unsupported")
                width, height = candidate.size
                if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                    raise ValueError("vision content dimensions are unsupported")
                candidate.verify()
            with image_module.open(io.BytesIO(payload)) as decoded:
                if decoded.format != _MEDIA_FORMATS[media_type]:
                    raise ValueError("vision content format changed while decoding")
                return decoded.convert("RGB").copy()
    except (OSError, ValueError, Warning) as exc:
        raise ValueError("vision content could not be decoded safely") from exc


def _record_material(
    record: EvaluationRecord,
) -> tuple[str, EvaluationInputPart, str]:
    parts = getattr(record, "input_parts", ())
    if not isinstance(parts, tuple) or not parts:
        raise ValueError("vision-text records require structured input_parts")
    prompt_parts = [
        part
        for part in parts
        if getattr(part, "kind", None) == "text"
        and getattr(part, "role", None) == "prompt"
    ]
    image_parts = [
        part
        for part in parts
        if getattr(part, "kind", None) == "content"
        and getattr(part, "role", None) == "image"
    ]
    if len(prompt_parts) != 1 or len(image_parts) != 1 or len(parts) != 2:
        raise ValueError(
            "vision-text records require exactly one prompt and one image part"
        )
    prompt_part = prompt_parts[0]
    image_part = image_parts[0]
    prompt = getattr(prompt_part, "text", None)
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("vision-text prompt must be non-empty text")
    if _sha256(prompt.encode("utf-8")) != getattr(prompt_part, "sha256", None):
        raise ValueError("vision-text prompt digest does not match")
    return prompt, image_part, record.input_sha256


def _validate_schedule_content(
    schedule: RuntimeBehavioralSchedule,
    *,
    content_store: Path,
) -> None:
    """Authenticate and safely decode schedule-bound media without loading a model."""

    if schedule.task != "vision_text_generation":
        raise ValueError("hf_vision_text requires vision_text_generation")
    observed: dict[str, tuple[str, int, str]] = {}
    total_bytes = 0
    total_pixels = 0
    for record in schedule.records:
        try:
            _prompt, image_part, _input_digest = _record_material(record)
            content_id = image_part.content_id
            media_type = image_part.media_type
            byte_length = image_part.byte_length
            digest = image_part.sha256
            if (
                not isinstance(content_id, str)
                or not isinstance(media_type, str)
                or isinstance(byte_length, bool)
                or not isinstance(byte_length, int)
                or not isinstance(digest, str)
            ):
                raise ValueError("vision content binding is incomplete")
            binding = (media_type, byte_length, digest)
            prior = observed.get(content_id)
            if prior is not None:
                if prior != binding:
                    raise ValueError(
                        "vision content_id has conflicting authenticated bindings"
                    )
                continue
            if len(observed) >= _MAX_UNIQUE_IMAGES:
                raise ValueError("vision schedule exceeds the unique image limit")
            total_bytes += byte_length
            if total_bytes > _MAX_TOTAL_IMAGE_BYTES:
                raise ValueError("vision schedule exceeds the total image byte limit")
            payload = _read_content_bytes(
                content_store,
                content_id=content_id,
                expected_sha256=digest,
                expected_byte_length=byte_length,
            )
            image = _decode_image(payload, media_type=media_type)
            close = getattr(image, "close", None)
            if not callable(close):
                raise ValueError("decoded vision content cannot be safely closed")
            try:
                size = getattr(image, "size", None)
                if (
                    not isinstance(size, tuple)
                    or len(size) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in size
                    )
                ):
                    raise ValueError("decoded vision content dimensions are invalid")
                total_pixels += size[0] * size[1]
                if total_pixels > _MAX_TOTAL_IMAGE_PIXELS:
                    raise ValueError(
                        "vision schedule exceeds the total decoded pixel limit"
                    )
            finally:
                close()
            observed[content_id] = binding
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            content_label = getattr(
                next(
                    (
                        part
                        for part in getattr(record, "input_parts", ())
                        if getattr(part, "kind", None) == "content"
                    ),
                    None,
                ),
                "content_id",
                "unknown",
            )
            raise ValueError(
                f"record {record.record_id!r} content {content_label!r}: {exc}"
            ) from exc


def _model_device(model: object) -> object:
    parameters = getattr(model, "parameters", None)
    buffers = getattr(model, "buffers", None)
    if not callable(parameters) or not callable(buffers):
        raise RuntimeError("vision-text model does not expose execution tensors")
    tensors = tuple(parameters()) or tuple(buffers())
    if not tensors:
        raise RuntimeError("vision-text model has no execution tensors")
    return tensors[0].device


def _require_eval_mode(model: object) -> None:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise RuntimeError("vision-text model does not expose module state")
    observed = tuple(modules())
    if not observed or any(
        getattr(module, "training", None) is not False for module in observed
    ):
        raise RuntimeError("vision-text model must remain in evaluation mode")


def _deadline_criterion(stopping_base: type, *, deadline: float) -> object:
    class _Deadline(stopping_base):
        def __call__(self, *args: object, **kwargs: object) -> bool:
            if time.monotonic() >= deadline:
                raise TimeoutError("vision-text generation timed out")
            return False

    return _Deadline()


@dataclass(frozen=True)
class HFVisionTextScorer:
    """Deterministic exact-match scorer over authenticated image/text records."""

    model: object = field(repr=False, compare=False)
    processor: object = field(repr=False, compare=False)
    content_store: Path = field(repr=False, compare=False)
    artifact_identity_sha256: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.artifact_identity_sha256) is None:
            raise ValueError("artifact_identity_sha256 must be a sha256 digest")
        content_store = Path(self.content_store)
        if not content_store.is_absolute():
            raise ValueError("content_store must be absolute")
        object.__setattr__(self, "content_store", content_store)

    def __call__(
        self, batch: EvaluationBatch, settings: RuntimeExecutionSettings
    ) -> ScoringObservation:
        if getattr(batch, "task", None) != "vision_text_generation":
            raise ValueError("hf_vision_text requires vision_text_generation")
        if batch.metric != "exact_match":
            raise ValueError("hf_vision_text currently supports exact_match only")
        if settings.batch_size != 1:
            raise ValueError("hf_vision_text currently requires batch_size=1")
        torch = importlib.import_module("torch")
        stopping_module = importlib.import_module(
            "transformers.generation.stopping_criteria"
        )
        stopping_base = getattr(stopping_module, "StoppingCriteria", None)
        stopping_list = getattr(stopping_module, "StoppingCriteriaList", None)
        if not isinstance(stopping_base, type) or not callable(stopping_list):
            raise RuntimeError("transformers stopping criteria are unavailable")
        processor_call = self.processor if callable(self.processor) else None
        apply_template = getattr(self.processor, "apply_chat_template", None)
        decoder = getattr(self.processor, "batch_decode", None)
        if (
            not callable(processor_call)
            or not callable(apply_template)
            or not callable(decoder)
        ):
            raise RuntimeError("vision-text processor APIs are unavailable")
        generate = getattr(self.model, "generate", None)
        if not callable(generate):
            raise RuntimeError("vision-text model generation API is unavailable")
        _require_eval_mode(self.model)
        device = _model_device(self.model)
        records: list[RuntimeScoringRecord] = []
        deterministic_enabled = bool(torch.are_deterministic_algorithms_enabled())
        deterministic_warn_only = bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            with torch.random.fork_rng(), torch.inference_mode():
                torch.manual_seed(settings.seed)
                if getattr(device, "type", None) == "cuda":
                    torch.cuda.manual_seed_all(settings.seed)
                for record in batch.records:
                    prompt, image_part, input_sha256 = _record_material(record)
                    content_id = getattr(image_part, "content_id", None)
                    media_type = getattr(image_part, "media_type", None)
                    content_sha256 = getattr(image_part, "sha256", None)
                    byte_length = getattr(image_part, "byte_length", None)
                    if not isinstance(content_id, str) or not isinstance(
                        media_type, str
                    ):
                        raise ValueError("vision content binding is incomplete")
                    payload = _read_content_bytes(
                        self.content_store,
                        content_id=content_id,
                        expected_sha256=cast(str, content_sha256),
                        expected_byte_length=cast(int, byte_length),
                    )
                    image = _decode_image(payload, media_type=media_type)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    rendered = apply_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if not isinstance(rendered, str) or not rendered:
                        raise RuntimeError("vision-text chat template returned no text")
                    try:
                        encoded = processor_call(
                            text=rendered,
                            images=image,
                            return_tensors="pt",
                            truncation=True,
                            max_length=settings.context_length,
                        )
                    finally:
                        close_image = getattr(image, "close", None)
                        if callable(close_image):
                            close_image()
                    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                        raise RuntimeError(
                            "vision-text processor returned invalid inputs"
                        )
                    model_inputs = {
                        key: value.to(device)
                        for key, value in encoded.items()
                        if hasattr(value, "to")
                    }
                    input_ids = model_inputs.get("input_ids")
                    if (
                        input_ids is None
                        or getattr(input_ids, "ndim", None) != 2
                        or input_ids.shape[0] != 1
                    ):
                        raise RuntimeError("vision-text input_ids are invalid")
                    deadline = time.monotonic() + settings.timeout_seconds

                    generated = generate(
                        **model_inputs,
                        do_sample=False,
                        max_new_tokens=settings.max_output_tokens,
                        stopping_criteria=stopping_list(
                            [_deadline_criterion(stopping_base, deadline=deadline)]
                        ),
                        use_cache=False,
                    )
                    if time.monotonic() >= deadline:
                        raise TimeoutError("vision-text generation timed out")
                    sequences = getattr(generated, "sequences", generated)
                    if getattr(sequences, "ndim", None) != 2 or sequences.shape[0] != 1:
                        raise RuntimeError(
                            "vision-text generation returned invalid IDs"
                        )
                    prompt_length = int(input_ids.shape[1])
                    if sequences.shape[1] >= prompt_length and bool(
                        torch.equal(sequences[:, :prompt_length], input_ids)
                    ):
                        sequences = sequences[:, prompt_length:]
                    decoded = decoder(
                        sequences,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if not isinstance(decoded, list) or len(decoded) != 1:
                        raise RuntimeError(
                            "vision-text decoder returned invalid output"
                        )
                    output_text = exact_match_output_text(decoded[0])
                    records.append(
                        RuntimeScoringRecord(
                            record_id=record.record_id,
                            input_sha256=input_sha256,
                            status="ok",
                            output_text=output_text,
                            output_sha256=_sha256(output_text.encode("utf-8")),
                        )
                    )
        finally:
            torch.use_deterministic_algorithms(
                deterministic_enabled,
                warn_only=deterministic_warn_only,
            )
        frozen = tuple(records)
        return ScoringObservation(
            provider_name="hf_vision_text",
            artifact_identity_sha256=self.artifact_identity_sha256,
            schedule_sha256=batch.schedule_sha256,
            records=frozen,
            aggregate_source_sha256=runtime_scoring_records_sha256(
                [asdict(record) for record in frozen]
            ),
        )


@dataclass(frozen=True)
class _ReceiptProvenance:
    backend: RuntimeBackendIdentity
    capabilities: RuntimeProviderCapabilities
    artifact_identity: HFSnapshotArtifactIdentity
    execution_settings: RuntimeExecutionSettings
    device: RuntimeDeviceFacts
    outer_image_digest: str


@dataclass
class _VisionTextSession:
    scorer: HFVisionTextScorer
    provenance: _ReceiptProvenance
    binding_check: object = field(repr=False)
    _latest_observation_sha256: str | None = field(default=None, init=False)
    _closed: bool = field(default=False, init=False)

    def score(self, batch: EvaluationBatch) -> ScoringObservation:
        if self._closed:
            raise RuntimeError("vision-text session is closed")
        if not callable(self.binding_check):
            raise RuntimeError("vision-text session binding check is unavailable")
        self.binding_check()
        try:
            observation = self.scorer(batch, self.provenance.execution_settings)
        finally:
            self.binding_check()
        self._latest_observation_sha256 = _sha256(
            encode_scoring_observation(observation)
        )
        return observation

    def runtime_receipt(self) -> RuntimeProviderReceipt:
        if self._closed:
            raise RuntimeError("vision-text session is closed")
        if self._latest_observation_sha256 is None:
            raise RuntimeError("runtime receipt is unavailable before scoring")
        return RuntimeProviderReceipt(
            plugin=RuntimeProviderPluginIdentity(
                name="hf_vision_text",
                distribution="invarlock-runtime-hf-vision-text",
                distribution_version=ADDIN_VERSION,
            ),
            backend=self.provenance.backend,
            capabilities=self.provenance.capabilities,
            artifact_identity=self.provenance.artifact_identity,
            execution_settings=self.provenance.execution_settings,
            device=self.provenance.device,
            outer_image_digest=self.provenance.outer_image_digest,
            scoring_observation_sha256=self._latest_observation_sha256,
        )

    def close(self) -> None:
        self._closed = True


def _module_file_sha256(module: object, *, label: str) -> str:
    path_value = getattr(module, "__file__", None)
    if not isinstance(path_value, str):
        raise RuntimeError(f"{label} module identity is unavailable")
    path = Path(path_value)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label} module identity is unavailable") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"{label} module identity is unavailable")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 64 * 1024):
            digest.update(chunk)
        if _stat_identity(os.fstat(descriptor)) != _stat_identity(opened):
            raise RuntimeError(f"{label} module changed while being identified")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _backend_identity() -> RuntimeBackendIdentity:
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    pillow = importlib.import_module("PIL")
    versions = {
        "pillow": str(getattr(pillow, "__version__", "")),
        "torch": str(getattr(torch, "__version__", "")),
        "transformers": str(getattr(transformers, "__version__", "")),
    }
    if not all(versions.values()):
        raise RuntimeError("vision-text backend versions are unavailable")
    modules = {
        "pillow": _module_file_sha256(pillow, label="Pillow"),
        "torch": _module_file_sha256(torch, label="torch"),
        "transformers": _module_file_sha256(transformers, label="transformers"),
    }
    return RuntimeBackendIdentity(
        name="huggingface-vision-text",
        version=";".join(f"{name}={versions[name]}" for name in sorted(versions)),
        source_sha256=None,
        binary_sha256=_sha256(
            json.dumps(modules, sort_keys=True, separators=(",", ":")).encode()
        ),
        build_sha256=None,
    )


def _device_facts(model: object, *, expected_kind: str) -> RuntimeDeviceFacts:
    device = _model_device(model)
    kind = str(getattr(device, "type", device))
    if kind != expected_kind:
        raise ValueError("vision-text model device does not match selected device")
    if kind == "cpu":
        return RuntimeDeviceFacts(
            device_kind="cpu",
            device_name=platform.processor() or platform.machine() or "CPU",
        )
    torch = importlib.import_module("torch")
    index = getattr(device, "index", None)
    name = torch.cuda.get_device_name(index)
    major, minor = torch.cuda.get_device_capability(index)
    driver_getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    driver_version = str(driver_getter()) if callable(driver_getter) else None
    return RuntimeDeviceFacts(
        device_kind="cuda",
        device_name=str(name),
        compute_capability=f"{major}.{minor}",
        driver_version=driver_version,
        cuda_runtime_version=str(torch.version.cuda),
    )


def _require_runtime_boundary(context: RuntimeExecutionContext) -> None:
    if context.allow_network or network_allowed():
        raise ValueError("hf_vision_text execution must be offline")
    if remote_code_allowed() or third_party_plugins_allowed():
        raise ValueError("hf_vision_text execution forbids remote code and plugins")
    if not strict_container_boundary_present():
        raise ValueError("hf_vision_text requires an authenticated container boundary")
    digest = context.container_image_digest
    if digest is None or resolve_runtime_image_digest() != digest:
        raise ValueError("hf_vision_text runtime image digest does not match")
    image = resolve_runtime_image()
    if image != digest and not image.endswith("@" + digest):
        raise ValueError("hf_vision_text runtime image reference does not bind digest")


class HFVisionTextProvider:
    """First-party optional provider for bounded image-and-text generation."""

    name = "hf_vision_text"
    abi_version = INVARLOCK_RUNTIME_PROVIDER_ABI

    def validate_config(self, spec: ModelRuntimeSpec) -> None:
        if spec.provider_name != self.name:
            raise ValueError(f"provider_name must be {self.name!r}")
        unknown = set(spec.settings) - _ALLOWED_SETTINGS
        if unknown:
            raise ValueError(
                "unsupported hf_vision_text setting(s): " + ", ".join(sorted(unknown))
            )
        missing = _REQUIRED_SETTINGS - set(spec.settings)
        if missing:
            raise ValueError(
                "missing hf_vision_text setting(s): " + ", ".join(sorted(missing))
            )
        for name in (
            "checkpoint_tree_sha256",
            "processor_metadata_sha256",
            "tokenizer_metadata_sha256",
        ):
            _required_digest(spec.settings, name)
        _optional_text(spec.settings, "immutable_revision")
        for name in (
            "batch_size",
            "context_length",
            "max_output_tokens",
            "timeout_seconds",
        ):
            _required_integer(spec.settings, name, positive=True)
        _required_integer(spec.settings, "seed", positive=False)
        if spec.settings["batch_size"] != 1:
            raise ValueError("hf_vision_text currently requires batch_size=1")
        if spec.settings["offline"] is not True:
            raise ValueError("hf_vision_text requires offline=true")

    def capabilities(self) -> RuntimeProviderCapabilities:
        return RuntimeProviderCapabilities(
            provider_name=self.name,
            artifact_formats=("hf_snapshot",),
            tasks=("vision_text_generation",),
            metrics=("exact_match",),
            execution_modes=("container",),
            required_extra=None,
            required_image=None,
        )

    def identify_artifact(self, spec: ModelRuntimeSpec) -> HFSnapshotArtifactIdentity:
        self.validate_config(spec)
        return HFSnapshotArtifactIdentity(
            model_id=spec.model_id,
            immutable_revision=_optional_text(spec.settings, "immutable_revision"),
            checkpoint_tree_sha256=_required_digest(
                spec.settings, "checkpoint_tree_sha256"
            ),
            tokenizer_metadata_sha256=_required_digest(
                spec.settings, "tokenizer_metadata_sha256"
            ),
        )

    def authenticate_artifact(
        self, spec: ModelRuntimeSpec, artifact_path: Path
    ) -> HFSnapshotArtifactIdentity:
        """Authenticate local checkpoint bytes without loading the model."""

        identity = self.identify_artifact(spec)
        try:
            observed_tree = checkpoint_tree_sha256(artifact_path).removeprefix(
                "sha256:"
            )
        except (CheckpointIdentityError, OSError) as exc:
            raise ValueError(
                "hf_vision_text checkpoint could not be authenticated"
            ) from exc
        if observed_tree != identity.checkpoint_tree_sha256:
            raise ValueError("hf_vision_text checkpoint tree digest does not match")
        return identity

    def validate_evaluation_inputs(
        self,
        spec: ModelRuntimeSpec,
        resources: RuntimeArtifactResources,
        schedule: RuntimeBehavioralSchedule,
    ) -> None:
        """Authenticate schedule-bound media before any model or GPU initialization."""

        self.validate_config(spec)
        resources.require_support_names(frozenset({"content_store"}))
        content_store = resources.support_path("content_store")
        if not content_store.is_dir():
            raise ValueError("hf_vision_text content directory is unavailable")
        _validate_schedule_content(schedule, content_store=content_store)

    def prepare_execution(
        self, spec: ModelRuntimeSpec, resources: RuntimeArtifactResources
    ) -> RuntimeExecutionContext:
        self.validate_config(spec)
        resources.require_support_names(frozenset({"content_store"}))
        checkpoint = resources.primary_path()
        content_store = resources.support_path("content_store")
        if not checkpoint.is_dir() or not content_store.is_dir():
            raise ValueError(
                "hf_vision_text requires checkpoint and content directories"
            )
        identity = self.authenticate_artifact(spec, checkpoint)
        transformers = importlib.import_module("transformers")
        processor_loader = getattr(transformers, "AutoProcessor", None)
        processor_from_pretrained = getattr(processor_loader, "from_pretrained", None)
        if not callable(processor_from_pretrained):
            raise RuntimeError("transformers AutoProcessor is unavailable")
        model_from_pretrained = None
        for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            loader = getattr(transformers, name, None)
            candidate = getattr(loader, "from_pretrained", None)
            if callable(candidate):
                model_from_pretrained = candidate
                break
        if model_from_pretrained is None:
            raise RuntimeError("transformers vision-text auto model is unavailable")
        processor = processor_from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=False
        )
        model = load_hf_model_with_strict_loading_info(
            model_from_pretrained,
            checkpoint,
        )
        move = getattr(model, "to", None)
        evaluate = getattr(model, "eval", None)
        if not callable(move) or not callable(evaluate):
            raise RuntimeError("loaded vision-text model APIs are unavailable")
        model = move(resources.device_kind)
        evaluate = getattr(model, "eval", None)
        assert callable(evaluate)
        evaluate()
        tokenizer = getattr(processor, "tokenizer", None)
        if (
            hf_tokenizer_contract_sha256(tokenizer)
            != identity.tokenizer_metadata_sha256
        ):
            raise ValueError("vision-text tokenizer contract does not match")
        if processor_contract_sha256(processor) != _required_digest(
            spec.settings, "processor_metadata_sha256"
        ):
            raise ValueError("vision-text processor contract does not match")
        require_loaded_hf_checkpoint_binding(
            spec=spec,
            identity=identity,
            model=model,
            tokenizer=tokenizer,
            checkpoint=checkpoint,
        )
        try:
            final_tree = checkpoint_tree_sha256(checkpoint).removeprefix("sha256:")
        except (CheckpointIdentityError, OSError) as exc:
            raise ValueError(
                "hf_vision_text checkpoint could not be reauthenticated"
            ) from exc
        if final_tree != identity.checkpoint_tree_sha256:
            raise RuntimeError("hf_vision_text checkpoint changed during loading")
        identity_sha = artifact_identity_sha256(identity)
        return RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=resources.container_image_digest,
            device_kind=resources.device_kind,
            artifact_identity_sha256=identity_sha,
            provider_state=(model, processor, checkpoint),
            scorer=HFVisionTextScorer(
                model=model,
                processor=processor,
                content_store=content_store,
                artifact_identity_sha256=identity_sha,
            ),
        )

    def open(
        self, spec: ModelRuntimeSpec, context: RuntimeExecutionContext
    ) -> _VisionTextSession:
        self.validate_config(spec)
        _require_runtime_boundary(context)
        if not isinstance(context.scorer, HFVisionTextScorer):
            raise ValueError("hf_vision_text requires its authenticated scorer")
        if (
            not isinstance(context.provider_state, tuple)
            or len(context.provider_state) != 3
        ):
            raise ValueError("hf_vision_text model and processor bindings are absent")
        identity = self.identify_artifact(spec)
        identity_sha = artifact_identity_sha256(identity)
        if context.artifact_identity_sha256 != identity_sha:
            raise ValueError("hf_vision_text runtime artifact identity does not match")
        model, processor, checkpoint = context.provider_state
        if (
            context.scorer.model is not model
            or context.scorer.processor is not processor
        ):
            raise ValueError("hf_vision_text scorer does not bind the loaded objects")
        _require_eval_mode(model)
        settings = runtime_execution_settings_from_mapping(
            spec.settings, allow_network=False
        )
        tokenizer = getattr(processor, "tokenizer", None)
        expected_processor_digest = _required_digest(
            spec.settings, "processor_metadata_sha256"
        )

        def binding_check() -> None:
            require_loaded_hf_checkpoint_binding(
                spec=spec,
                identity=identity,
                model=model,
                tokenizer=tokenizer,
                checkpoint=cast(Path, checkpoint),
            )
            if processor_contract_sha256(processor) != expected_processor_digest:
                raise ValueError("vision-text processor contract changed")

        image_digest = context.container_image_digest
        assert image_digest is not None
        return _VisionTextSession(
            scorer=context.scorer,
            binding_check=binding_check,
            provenance=_ReceiptProvenance(
                backend=_backend_identity(),
                capabilities=self.capabilities(),
                artifact_identity=identity,
                execution_settings=settings,
                device=_device_facts(model, expected_kind=context.device_kind),
                outer_image_digest=image_digest,
            ),
        )


__all__ = [
    "HFVisionTextProvider",
    "HFVisionTextScorer",
    "INVARLOCK_RUNTIME_PROVIDER_ABI",
    "processor_contract_sha256",
]
