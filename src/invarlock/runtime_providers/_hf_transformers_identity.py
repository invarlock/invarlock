"""Identity helpers for the Hugging Face runtime provider.

This internal module owns path-free tokenizer contracts and the installed
Transformers/Torch runtime identity. Keeping those concerns separate leaves the
provider module focused on validation, scoring, and session lifecycle.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
from collections.abc import Mapping
from pathlib import Path

from invarlock.core.runtime_provider import RuntimeBackendIdentity, RuntimeDeviceFacts
from invarlock.evidence_pack_json import StrictJsonError, read_regular_file_bytes

_LOCAL_VERSION_PATTERN = re.compile(
    r"^(?P<public>[^+]+)\+(?P<local>[a-z0-9]+(?:[._-][a-z0-9]+)*)$",
    re.IGNORECASE,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(encoded)


def _json_safe_tokenizer_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_tokenizer_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set):
        return [
            _json_safe_tokenizer_value(item)
            for item in sorted(value, key=lambda item: str(item))
        ]
    if isinstance(value, (list, tuple)):
        return [_json_safe_tokenizer_value(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe_tokenizer_value(to_dict())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return str(value)


def hf_tokenizer_contract_sha256(tokenizer: object) -> str:
    """Return a path-free digest of the live tokenizer behavior contract."""

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise RuntimeError("strict HF tokenizer does not expose a vocabulary")
    try:
        vocab = get_vocab()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("strict HF tokenizer vocabulary is unavailable") from exc
    if not isinstance(vocab, Mapping) or not vocab:
        raise RuntimeError("strict HF tokenizer vocabulary is unavailable")
    normalized_vocab: dict[str, int] = {}
    for token, token_id in vocab.items():
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise RuntimeError("strict HF tokenizer vocabulary is invalid")
        normalized_vocab[str(token)] = token_id

    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend_tokenizer, "to_str", None)
    backend_contract: object = None
    if callable(backend_to_str):
        try:
            serialized = backend_to_str()
            if not isinstance(serialized, str) or not serialized:
                raise ValueError
            backend_contract = json.loads(serialized)
            if isinstance(backend_contract, dict):
                # Fast-tokenizer calls persist request-local padding/truncation
                # state in the backend object. Scoring supplies these controls
                # explicitly, so they are not immutable tokenizer identity.
                backend_contract.pop("padding", None)
                backend_contract.pop("truncation", None)
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                "strict HF fast-tokenizer backend contract is unavailable"
            ) from exc

    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    try:
        added_vocab = get_added_vocab() if callable(get_added_vocab) else {}
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "strict HF tokenizer added vocabulary is unavailable"
        ) from exc
    if not isinstance(added_vocab, Mapping):
        raise RuntimeError("strict HF tokenizer added vocabulary is invalid")

    payload = {
        "added_vocab": _json_safe_tokenizer_value(added_vocab),
        "backend": backend_contract,
        "chat_template": _json_safe_tokenizer_value(
            getattr(tokenizer, "chat_template", None)
        ),
        "class": (
            f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}"
        ),
        "clean_up_tokenization_spaces": _json_safe_tokenizer_value(
            getattr(tokenizer, "clean_up_tokenization_spaces", None)
        ),
        "model_max_length": _json_safe_tokenizer_value(
            getattr(tokenizer, "model_max_length", None)
        ),
        "padding_side": _json_safe_tokenizer_value(
            getattr(tokenizer, "padding_side", None)
        ),
        "special_tokens": _json_safe_tokenizer_value(
            getattr(tokenizer, "special_tokens_map", {})
        ),
        "truncation_side": _json_safe_tokenizer_value(
            getattr(tokenizer, "truncation_side", None)
        ),
        "vocab": normalized_vocab,
    }
    return _canonical_sha256(payload)


def _regular_file_sha256(path: object, *, label: str) -> str:
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"{label} file identity is unavailable")
    try:
        payload = read_regular_file_bytes(Path(path), label=label)
    except (OSError, StrictJsonError) as exc:
        raise RuntimeError(f"{label} file identity is unavailable") from exc
    return _sha256(payload)


def _distribution_identity(name: str) -> dict[str, str]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required {name} distribution is not installed") from exc
    metadata_text = distribution.read_text("METADATA")
    record_text = distribution.read_text("RECORD")
    if metadata_text is None or record_text is None:
        raise RuntimeError(
            f"installed {name} distribution lacks METADATA or RECORD identity"
        )
    version = distribution.version
    if not version:
        raise RuntimeError(f"installed {name} distribution lacks a version")
    return {
        "metadata_sha256": _sha256(metadata_text.encode("utf-8")),
        "name": name,
        "record_sha256": _sha256(record_text.encode("utf-8")),
        "version": version,
    }


def _runtime_version_matches_distribution(
    imported_version: object, installed_version: str
) -> bool:
    """Match an exact public version while accounting for PEP 440 local labels."""

    def _split_local(value: object) -> tuple[str, str | None] | None:
        if not isinstance(value, str) or not value or value.strip() != value:
            return None
        if "+" not in value:
            return value, None
        match = _LOCAL_VERSION_PATTERN.fullmatch(value)
        if match is None:
            return None
        normalized_local = re.sub(r"[._-]+", ".", match.group("local")).lower()
        return match.group("public"), normalized_local

    imported = _split_local(imported_version)
    installed = _split_local(installed_version)
    if imported is None or installed is None or imported[0] != installed[0]:
        return False
    imported_local, installed_local = imported[1], installed[1]
    return (
        imported_local is None
        or installed_local is None
        or imported_local == installed_local
    )


def _installed_backend_identity(model: object) -> RuntimeBackendIdentity:
    """Bind the imported HF/Torch runtime to installed distribution material."""

    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    torch_c = importlib.import_module("torch._C")
    transformers_distribution = _distribution_identity("transformers")
    torch_distribution = _distribution_identity("torch")
    transformers_version = getattr(transformers, "__version__", None)
    torch_version = getattr(torch, "__version__", None)
    if not _runtime_version_matches_distribution(
        transformers_version, transformers_distribution["version"]
    ):
        raise RuntimeError(
            "imported transformers version does not match installed distribution"
        )
    if not _runtime_version_matches_distribution(
        torch_version, torch_distribution["version"]
    ):
        raise RuntimeError(
            "imported torch version does not match installed distribution"
        )

    model_module_name = type(model).__module__
    if not model_module_name or model_module_name == "builtins":
        raise RuntimeError("native HF model implementation identity is unavailable")
    model_module = importlib.import_module(model_module_name)
    transformers_source_sha256 = _regular_file_sha256(
        getattr(transformers, "__file__", None),
        label="transformers package source",
    )
    model_source_sha256 = _regular_file_sha256(
        getattr(model_module, "__file__", None),
        label="native HF model source",
    )
    torch_binary_sha256 = _regular_file_sha256(
        getattr(torch_c, "__file__", None),
        label="torch native extension",
    )
    torch_config = getattr(getattr(torch, "__config__", None), "show", None)
    if not callable(torch_config):
        raise RuntimeError("imported torch build configuration is unavailable")
    torch_build_config = torch_config()
    if not isinstance(torch_build_config, str) or not torch_build_config:
        raise RuntimeError("imported torch build configuration is unavailable")

    source_sha256 = _canonical_sha256(
        {
            "model_module": model_module_name,
            "model_source_sha256": model_source_sha256,
            "transformers": transformers_distribution,
            "transformers_source_sha256": transformers_source_sha256,
        }
    )
    build_sha256 = _canonical_sha256(
        {
            "model_module": model_module_name,
            "model_source_sha256": model_source_sha256,
            "torch": torch_distribution,
            "torch_binary_sha256": torch_binary_sha256,
            "torch_build_config_sha256": _sha256(torch_build_config.encode("utf-8")),
            "transformers": transformers_distribution,
            "transformers_source_sha256": transformers_source_sha256,
        }
    )
    return RuntimeBackendIdentity(
        name="transformers+torch",
        version=(
            f"transformers={transformers_distribution['version']};"
            f"torch={torch_distribution['version']}"
        ),
        source_sha256=source_sha256,
        binary_sha256=torch_binary_sha256,
        build_sha256=build_sha256,
    )


def _observed_device_facts(
    model: object, *, expected_device_kind: str
) -> RuntimeDeviceFacts:
    torch = importlib.import_module("torch")
    module_type = getattr(getattr(torch, "nn", None), "Module", None)
    if module_type is None or not isinstance(model, module_type):
        raise RuntimeError("native HF model must be a torch module")
    parameters = tuple(model.parameters())
    tensors = parameters or tuple(model.buffers())
    if not tensors:
        raise RuntimeError("native HF model has no tensors for device identity")
    devices = {(str(tensor.device.type), tensor.device.index) for tensor in tensors}
    if len(devices) != 1:
        raise RuntimeError(
            "strict hf_transformers receipt requires one model execution device"
        )
    device_kind, device_index = devices.pop()
    if device_kind != expected_device_kind:
        raise ValueError(
            "observed HF model device does not match the runtime execution context"
        )

    compute_capability: str | None = None
    driver_version: str | None = None
    if device_kind == "cuda":
        if not bool(torch.cuda.is_available()):
            raise RuntimeError("observed CUDA model device is unavailable")
        cuda_index = (
            int(torch.cuda.current_device())
            if device_index is None
            else int(device_index)
        )
        properties = torch.cuda.get_device_properties(cuda_index)
        device_name = str(properties.name)
        compute_capability = f"{int(properties.major)}.{int(properties.minor)}"
        driver_getter = getattr(torch._C, "_cuda_getDriverVersion", None)
        if callable(driver_getter):
            driver_version = str(driver_getter())
    elif device_kind == "cpu":
        cpu_name = platform.processor().strip() or platform.machine().strip()
        device_name = f"CPU {cpu_name}" if cpu_name else "CPU"
    elif device_kind == "mps":
        device_name = "Apple Metal Performance Shaders"
    else:
        device_name = device_kind
    return RuntimeDeviceFacts(
        device_kind=device_kind,
        device_name=device_name,
        compute_capability=compute_capability,
        driver_version=driver_version,
    )


__all__ = [
    "_installed_backend_identity",
    "_json_safe_tokenizer_value",
    "_observed_device_facts",
    "_sha256",
    "hf_tokenizer_contract_sha256",
]
