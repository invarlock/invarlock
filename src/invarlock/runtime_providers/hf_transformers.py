"""Built-in Hugging Face provider for authenticated local checkpoints.

The provider loads a resolved local checkpoint, records its model and tokenizer
identity, and emits per-record scoring facts for independent verification.
Optional backend imports remain lazy so the core package stays Torch-free.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from invarlock import __version__ as INVARLOCK_VERSION
from invarlock.core.checkpoint_identity import (
    CheckpointIdentityError,
    checkpoint_tree_sha256,
)
from invarlock.core.runtime_provider import (
    INVARLOCK_RUNTIME_PROVIDER_ABI,
    EvaluationBatch,
    HFSnapshotArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeBackendIdentity,
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
)
from invarlock.core.runtime_provider.behavioral_observation import (
    runtime_scoring_records_sha256,
)
from invarlock.core.runtime_provider.types import (
    JSONScalar,
    RuntimeScorer,
    evaluation_input_parts_sha256,
)
from invarlock.runtime_providers._hf_safetensors_identity import (
    HFSafetensorsIdentityError,
    safetensors_storage_keys,
)
from invarlock.runtime_providers._hf_transformers_identity import (
    _installed_backend_identity,
    _json_safe_tokenizer_value,
    _observed_device_facts,
    _sha256,
    hf_tokenizer_contract_sha256,
)
from invarlock.runtime_security_helpers import (
    network_allowed,
    remote_code_allowed,
    resolve_runtime_image,
    resolve_runtime_image_digest,
    strict_container_boundary_present,
    third_party_plugins_allowed,
)

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
_REQUIRED_RECEIPT_SETTINGS = frozenset(
    {
        "batch_size",
        "context_length",
        "max_output_tokens",
        "offline",
        "seed",
        "timeout_seconds",
    }
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


def _strict_execution_settings(spec: ModelRuntimeSpec) -> RuntimeExecutionSettings:
    missing = _REQUIRED_RECEIPT_SETTINGS - set(spec.settings)
    if missing:
        rendered = ", ".join(sorted(missing))
        raise ValueError(
            "strict hf_transformers receipt requires setting(s): " + rendered
        )
    if spec.settings.get("offline") is not True:
        raise ValueError("strict hf_transformers receipt requires offline=true")
    return RuntimeExecutionSettings(
        seed=_required_integer(spec.settings, "seed", positive=False),
        context_length=_required_integer(
            spec.settings, "context_length", positive=True
        ),
        batch_size=_required_integer(spec.settings, "batch_size", positive=True),
        max_output_tokens=_required_integer(
            spec.settings, "max_output_tokens", positive=True
        ),
        timeout_seconds=_required_integer(
            spec.settings, "timeout_seconds", positive=True
        ),
        allow_network=False,
    )


def _image_ref_matches_digest(image_ref: str, image_digest: str) -> bool:
    if image_ref == image_digest:
        return True
    repository, separator, digest = image_ref.rpartition("@")
    return bool(repository and separator and digest == image_digest)


def _require_strict_runtime_boundary(context: RuntimeExecutionContext) -> None:
    if context.allow_network:
        raise ValueError("strict hf_transformers execution must disable network")
    if network_allowed():
        raise ValueError("strict hf_transformers execution must be offline")
    if remote_code_allowed():
        raise ValueError("strict hf_transformers execution must disable remote code")
    if third_party_plugins_allowed():
        raise ValueError(
            "strict hf_transformers execution must disable third-party plugins"
        )
    if not strict_container_boundary_present():
        raise ValueError(
            "strict hf_transformers execution requires the authenticated "
            "container boundary"
        )
    image_digest = context.container_image_digest
    if image_digest is None:
        raise ValueError(
            "strict hf_transformers execution requires a pinned outer container image"
        )
    if resolve_runtime_image_digest() != image_digest:
        raise ValueError(
            "strict hf_transformers runtime image digest does not match the "
            "container context"
        )
    if not _image_ref_matches_digest(resolve_runtime_image(), image_digest):
        raise ValueError(
            "strict hf_transformers runtime image reference must embed the exact digest"
        )


def _live_tensor_candidates(key: str, prefix: str | None) -> tuple[str, ...]:
    candidates = [key]
    if prefix is not None:
        prefix_marker = f"{prefix}."
        if not key.startswith(prefix_marker):
            candidates.append(f"{prefix}.{key}")
        else:
            suffix = key[len(prefix_marker) :]
            candidates.extend(
                f"{prefix}.{component}.{suffix}"
                for component in ("language_model", "text_model")
            )
    return tuple(dict.fromkeys(candidates))


_QWEN3_5_NON_EXECUTING_MTP_KEYS = frozenset(
    {
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
)


def _authoritative_checkpoint_key_targets(
    authenticated_keys: set[str],
    *,
    live_state: Mapping[str, object],
    model: object,
) -> dict[str, str]:
    """Apply the exact native Transformers renames used during model loading."""

    model_module = model.__class__.__module__
    if not model_module.startswith("transformers.models."):
        return {key: key for key in authenticated_keys}
    try:
        conversion_mapping = importlib.import_module("transformers.conversion_mapping")
        core_loading = importlib.import_module("transformers.core_model_loading")
        get_mapping = conversion_mapping.get_model_conversion_mapping
        rename_source_key = core_loading.rename_source_key
        weight_renaming = core_loading.WeightRenaming
        weight_converter = core_loading.WeightConverter
        if not callable(get_mapping) or not callable(rename_source_key):
            raise TypeError
        conversions = get_mapping(model)
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "strict HF native checkpoint conversion metadata is unavailable"
        ) from exc
    if not isinstance(conversions, list) or any(
        not isinstance(item, (weight_renaming, weight_converter))
        for item in conversions
    ):
        raise RuntimeError("strict HF native checkpoint conversion metadata is invalid")
    if any(isinstance(item, weight_converter) for item in conversions):
        raise ValueError("strict HF checkpoint requires unsupported tensor conversion")
    renamings = [item for item in conversions if isinstance(item, weight_renaming)]
    targets: dict[str, str] = {}
    sources_by_target: dict[str, str] = {}
    live_snapshot = dict(live_state)
    prefix = getattr(model, "base_model_prefix", None)
    base_model_prefix = prefix if isinstance(prefix, str) else None
    for source in sorted(authenticated_keys):
        try:
            target, converter_pattern = rename_source_key(
                source,
                renamings,
                [],
                base_model_prefix,
                live_snapshot,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "strict HF native checkpoint key conversion failed"
            ) from exc
        if (
            not isinstance(target, str)
            or not target
            or target != target.strip()
            or converter_pattern is not None
        ):
            raise RuntimeError(
                "strict HF native checkpoint conversion metadata is invalid"
            )
        prior_source = sources_by_target.get(target)
        if prior_source is not None and prior_source != source:
            raise ValueError("strict HF checkpoint key conversion is not one-to-one")
        sources_by_target[target] = source
        targets[source] = target
    return targets


def _qwen3_5_non_executing_checkpoint_keys(
    authenticated_keys: set[str],
    *,
    live_state: Mapping[str, object],
    model: object,
) -> set[str]:
    """Recognize the exact native Qwen3.5/3.6 inference-unused MTP inventory."""

    mtp_keys = {key for key in authenticated_keys if key.startswith("mtp.")}
    if not mtp_keys:
        return set()
    try:
        qwen_module = importlib.import_module(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
        expected_causal_class = qwen_module.Qwen3_5ForCausalLM
        expected_multimodal_class = qwen_module.Qwen3_5ForConditionalGeneration
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "strict HF native Qwen3.5 compatibility profile is unavailable"
        ) from exc
    model_class = model.__class__
    config = getattr(model, "config", None)
    accepted_native_profiles = (
        (
            expected_causal_class,
            "qwen3_5_text",
            [r"^mtp.*", r"^model.visual.*"],
        ),
        (
            expected_multimodal_class,
            "qwen3_5",
            [r"^mtp.*"],
        ),
    )
    if (
        not any(
            model_class is native_class
            and getattr(config, "model_type", None) == model_type
            and getattr(model_class, "_keys_to_ignore_on_load_unexpected", None)
            == ignored_keys
            for native_class, model_type, ignored_keys in accepted_native_profiles
        )
        or mtp_keys != _QWEN3_5_NON_EXECUTING_MTP_KEYS
    ):
        raise ValueError(
            "strict HF checkpoint contains an unsupported non-executing "
            "tensor inventory"
        )

    def names_from(method_name: str) -> tuple[str, ...]:
        method = getattr(model, method_name, None)
        if not callable(method):
            raise RuntimeError("strict HF native model state is unavailable")
        try:
            entries = tuple(method())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("strict HF native model state is unavailable") from exc
        names: list[str] = []
        for entry in entries:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
            ):
                raise RuntimeError("strict HF native model state is unavailable")
            names.append(entry[0])
        return tuple(names)

    live_names = set(live_state)
    live_names.update(names_from("named_parameters"))
    live_names.update(names_from("named_buffers"))
    live_names.update(names_from("named_modules"))
    if any(name == "mtp" or name.startswith("mtp.") for name in live_names):
        raise ValueError(
            "strict HF non-executing checkpoint tensors overlap live model state"
        )
    return set(mtp_keys)


def _qwen3_5_native_float32_to_bfloat16_keys(
    model: object,
    *,
    authenticated_config: object | None,
) -> set[str] | None:
    """Return the exact native mixed-dtype inventory cast by Transformers."""

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if (
        getattr(config, "model_type", None) != "qwen3_5"
        or getattr(text_config, "model_type", None) != "qwen3_5_text"
    ):
        return None
    try:
        qwen_module = importlib.import_module(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
        expected_class = qwen_module.Qwen3_5ForConditionalGeneration
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "strict HF native Qwen3.5 compatibility profile is unavailable"
        ) from exc
    if model.__class__ is not expected_class:
        raise ValueError("strict HF native Qwen3.5 model class is unsupported")
    if authenticated_config is None:
        raise RuntimeError(
            "strict HF native Qwen3.5 authenticated configuration is unavailable"
        )
    authenticated_text_config = getattr(authenticated_config, "text_config", None)
    if (
        authenticated_config.__class__ is not config.__class__
        or authenticated_text_config.__class__ is not text_config.__class__
        or getattr(authenticated_config, "model_type", None) != "qwen3_5"
        or getattr(authenticated_text_config, "model_type", None) != "qwen3_5_text"
    ):
        raise ValueError(
            "strict HF native Qwen3.5 authenticated configuration is unsupported"
        )
    authenticated_dtype_value = getattr(authenticated_config, "dtype", None)
    authenticated_dtype = str(authenticated_dtype_value).removeprefix("torch.")
    authenticated_text_dtype = str(
        getattr(authenticated_text_config, "dtype", None)
    ).removeprefix("torch.")
    live_dtype = str(getattr(config, "dtype", None)).removeprefix("torch.")
    live_text_dtype = str(getattr(text_config, "dtype", None)).removeprefix("torch.")
    if (
        authenticated_dtype_value is not None and authenticated_dtype != "bfloat16"
    ) or authenticated_text_dtype != "bfloat16":
        if live_dtype == "bfloat16" or live_text_dtype == "bfloat16":
            raise ValueError(
                "strict HF native Qwen3.5 bfloat16 materialization is not "
                "authorized by the checkpoint configuration"
            )
        return None
    if live_dtype != "bfloat16" or live_text_dtype != "bfloat16":
        raise ValueError(
            "strict HF native Qwen3.5 bfloat16 materialization was not preserved"
        )
    if (
        getattr(authenticated_config, "quantization_config", None) is not None
        or getattr(config, "quantization_config", None) is not None
        or bool(getattr(model, "is_quantized", False))
        or getattr(model, "hf_quantizer", None) is not None
    ):
        raise ValueError(
            "strict HF native Qwen3.5 dtype conversion requires an "
            "unquantized checkpoint"
        )
    dtype_plan_loader = getattr(model, "_get_dtype_plan", None)
    if not callable(dtype_plan_loader):
        raise RuntimeError("strict HF native model dtype plan is unavailable")
    try:
        dtype_plan = dtype_plan_loader(getattr(authenticated_config, "dtype", None))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("strict HF native model dtype plan is unavailable") from exc
    if not isinstance(dtype_plan, Mapping) or dtype_plan:
        raise ValueError(
            "strict HF native Qwen3.5 dtype conversion plan is unsupported"
        )
    layer_count = getattr(authenticated_text_config, "num_hidden_layers", None)
    layer_types = getattr(authenticated_text_config, "layer_types", None)
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
        or not isinstance(layer_types, (list, tuple))
        or len(layer_types) != layer_count
        or any(
            layer_type not in {"linear_attention", "full_attention"}
            for layer_type in layer_types
        )
    ):
        raise ValueError("strict HF native Qwen3.5 dtype conversion profile is invalid")
    return {
        f"model.language_model.layers.{index}.linear_attn.{suffix}"
        for index, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
        for suffix in ("A_log", "norm.weight")
    }


def _is_legacy_gpt2_causal_mask_key(
    key: str,
    *,
    model: object,
    prefix: str | None,
) -> bool:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "gpt2":
        return False
    positions = getattr(config, "max_position_embeddings", None)
    layers = getattr(config, "num_hidden_layers", None)
    if (
        isinstance(positions, bool)
        or not isinstance(positions, int)
        or positions <= 0
        or isinstance(layers, bool)
        or not isinstance(layers, int)
        or layers <= 0
    ):
        return False
    normalized_key = key
    if prefix is not None and key.startswith(f"{prefix}."):
        normalized_key = key[len(prefix) + 1 :]
    match = re.fullmatch(r"h\.(0|[1-9][0-9]*)\.attn\.bias", normalized_key)
    return match is not None and int(match.group(1)) < layers


def _is_authenticated_legacy_gpt2_causal_mask(
    key: str,
    tensor: object,
    *,
    model: object,
    prefix: str | None,
) -> bool:
    if not _is_legacy_gpt2_causal_mask_key(key, model=model, prefix=prefix):
        return False
    positions = getattr(
        getattr(model, "config", None),
        "max_position_embeddings",
        None,
    )
    if isinstance(positions, bool) or not isinstance(positions, int):
        return False
    shape = getattr(tensor, "shape", None)
    dtype = getattr(tensor, "dtype", None)
    new_ones = getattr(tensor, "new_ones", None)
    equal = getattr(tensor, "equal", None)
    if (
        not isinstance(shape, (list, tuple))
        or tuple(shape) != (1, 1, positions, positions)
        or str(dtype) != "torch.float32"
        or not callable(new_ones)
        or not callable(equal)
    ):
        return False
    expected = new_ones((1, 1, positions, positions)).tril()
    return bool(equal(expected))


def _tensor_storage_identity(value: object) -> tuple[object, ...] | None:
    try:
        detach = getattr(value, "detach", None)
        if not callable(detach):
            return None
        tensor = detach()
        pointer = tensor.untyped_storage().data_ptr()
        if not pointer:
            return None
        return (
            pointer,
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _tensors_share_exact_storage(left: object, right: object) -> bool:
    if left is right:
        return True
    left_identity = _tensor_storage_identity(left)
    return left_identity is not None and left_identity == _tensor_storage_identity(
        right
    )


def load_hf_model_with_strict_loading_info(
    loader: Callable[..., object],
    checkpoint: Path,
) -> object:
    """Load one local HF model and reject incomplete or ambiguous loader state."""

    loaded = loader(
        str(checkpoint),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
        dtype="auto",
    )
    if (
        not isinstance(loaded, tuple)
        or len(loaded) != 2
        or not isinstance(loaded[1], Mapping)
    ):
        raise RuntimeError("strict HF loader did not return loading information")
    model, loading_info = loaded
    fields: dict[str, tuple[object, ...]] = {}
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        value = loading_info.get(name)
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise RuntimeError("strict HF loader returned invalid loading information")
        fields[name] = tuple(value)
    prefix = getattr(model, "base_model_prefix", None)
    authenticated_prefix = (
        prefix if isinstance(prefix, str) and prefix.isidentifier() else None
    )
    unexpected = tuple(
        key
        for key in fields["unexpected_keys"]
        if not isinstance(key, str)
        or not _is_legacy_gpt2_causal_mask_key(
            key,
            model=model,
            prefix=authenticated_prefix,
        )
    )
    if (
        fields["missing_keys"]
        or unexpected
        or fields["mismatched_keys"]
        or fields["error_msgs"]
    ):
        raise ValueError(
            "strict HF checkpoint loading reported missing, unexpected, "
            "mismatched, or invalid model tensors"
        )
    return model


def _bind_authenticated_live_tensors(
    authenticated_keys: set[str],
    *,
    live_state: Mapping[str, object],
    model: object,
    prefix: str | None,
    authoritative_targets: Mapping[str, str],
) -> dict[str, tuple[tuple[str, object], ...]]:
    get_buffer = getattr(model, "get_buffer", None)
    bindings: dict[str, tuple[tuple[str, object], ...]] = {}
    source_by_live_name: dict[str, str] = {}
    for key in sorted(authenticated_keys):
        matches: list[tuple[str, object]] = []
        authoritative_target = authoritative_targets[key]
        candidates = (
            (authoritative_target,)
            if authoritative_target != key
            else _live_tensor_candidates(key, prefix)
        )
        for candidate in candidates:
            if candidate in live_state:
                matches.append((candidate, live_state[candidate]))
                continue
            if callable(get_buffer):
                try:
                    matches.append((candidate, get_buffer(candidate)))
                except (AttributeError, KeyError):
                    continue
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "strict HF native model buffer state is unavailable"
                    ) from exc
        if len(matches) > 1 and any(
            not _tensors_share_exact_storage(matches[0][1], candidate[1])
            for candidate in matches[1:]
        ):
            raise ValueError(
                "strict HF loaded model has an ambiguous authenticated "
                "checkpoint tensor mapping"
            )
        for live_name, _value in matches:
            prior_source = source_by_live_name.get(live_name)
            if prior_source is not None and prior_source != key:
                raise ValueError(
                    "strict HF checkpoint key conversion is not one-to-one"
                )
            source_by_live_name[live_name] = key
        bindings[key] = tuple(matches)
    return bindings


def _verify_authenticated_safetensors(
    checkpoint: Path,
    *,
    authenticated_keys: set[str],
    bindings: Mapping[str, tuple[tuple[str, object], ...]],
    model: object,
    prefix: str | None,
    safe_open: Callable[..., Any],
    non_executing_keys: set[str],
    authenticated_config: object | None,
) -> tuple[set[str], list[object]]:
    observed_keys: set[str] = set()
    authenticated_live_names: set[str] = set()
    authenticated_live_tensors: list[object] = []
    native_cast_profile = _qwen3_5_native_float32_to_bfloat16_keys(
        model,
        authenticated_config=authenticated_config,
    )
    allowed_native_casts = native_cast_profile or set()
    observed_native_casts: set[str] = set()
    shards = sorted(checkpoint.glob("*.safetensors"), key=lambda path: path.name)
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                observed_keys.add(key)
                bound_matches = bindings[key]
                if not bound_matches and key in non_executing_keys:
                    continue
                stored = handle.get_tensor(key)
                if not bound_matches:
                    if _is_authenticated_legacy_gpt2_causal_mask(
                        key,
                        stored,
                        model=model,
                        prefix=prefix,
                    ):
                        continue
                    raise ValueError(
                        "strict HF loaded model is missing authenticated "
                        "checkpoint tensors"
                    )
                live = bound_matches[0][1]
                detach = getattr(live, "detach", None)
                if not callable(detach):
                    raise RuntimeError(
                        "strict HF native model state contains a non-tensor value"
                    )
                live_cpu = detach().to(device="cpu").contiguous()
                shape_matches = tuple(stored.shape) == tuple(live_cpu.shape)
                exact_match = (
                    stored.dtype == live_cpu.dtype
                    and shape_matches
                    and bool(stored.equal(live_cpu))
                )
                stored_is_float = bool(
                    stored.is_floating_point() or stored.is_complex()
                )
                live_is_float = bool(
                    live_cpu.is_floating_point() or live_cpu.is_complex()
                )
                native_cast_match = (
                    native_cast_profile is not None
                    and key in allowed_native_casts
                    and str(stored.dtype) == "torch.float32"
                    and str(live_cpu.dtype) == "torch.bfloat16"
                    and shape_matches
                    and bool(stored.to(dtype=live_cpu.dtype).equal(live_cpu))
                )
                profile_exact_match = (
                    native_cast_profile is not None
                    and str(stored.dtype) == "torch.bfloat16"
                    and str(live_cpu.dtype) == "torch.bfloat16"
                    and shape_matches
                    and bool(stored.equal(live_cpu))
                )
                accepted_exact_match = (
                    profile_exact_match
                    if (
                        native_cast_profile is not None
                        and (stored_is_float or live_is_float)
                    )
                    else exact_match
                )
                if not accepted_exact_match and not native_cast_match:
                    raise ValueError(
                        "strict HF loaded model tensors do not match the "
                        "authenticated checkpoint"
                    )
                if native_cast_match:
                    observed_native_casts.add(key)
                authenticated_live_names.update(name for name, _value in bound_matches)
                authenticated_live_tensors.extend(
                    value for _name, value in bound_matches
                )
    if observed_keys != authenticated_keys:
        raise RuntimeError(
            "strict HF checkpoint tensor inventory changed during binding"
        )
    if observed_native_casts and observed_native_casts != allowed_native_casts:
        raise ValueError(
            "strict HF checkpoint native dtype conversion profile is incomplete"
        )
    return authenticated_live_names, authenticated_live_tensors


def _require_complete_live_state(
    live_state: Mapping[str, object],
    *,
    authenticated_names: set[str],
    authenticated_tensors: list[object],
) -> None:
    authenticated_objects = {id(value) for value in authenticated_tensors}
    authenticated_storage = {
        identity
        for value in authenticated_tensors
        if (identity := _tensor_storage_identity(value)) is not None
    }
    unauthenticated = []
    for key, value in live_state.items():
        if key in authenticated_names or id(value) in authenticated_objects:
            continue
        identity = _tensor_storage_identity(value)
        if identity is not None and identity in authenticated_storage:
            continue
        unauthenticated.append(key)
    if unauthenticated:
        raise ValueError(
            "strict HF loaded model contains unauthenticated live model tensors"
        )


def _require_safetensors_match(
    checkpoint: Path,
    *,
    model: object,
    authenticated_config: object | None = None,
) -> None:
    """Authenticate stored tensors and bind every live execution tensor exactly."""

    try:
        from safetensors import safe_open

    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "strict HF artifact binding requires the safetensors runtime"
        ) from exc

    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise RuntimeError("strict HF native model does not expose state_dict")
    try:
        live_state = state_dict()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("strict HF native model state is unavailable") from exc
    if not isinstance(live_state, Mapping):
        raise RuntimeError("strict HF native model state is unavailable")

    try:
        authenticated_keys = safetensors_storage_keys(checkpoint)
    except HFSafetensorsIdentityError as exc:
        raise RuntimeError(
            "strict HF checkpoint must use a canonical safetensors layout"
        ) from exc
    prefix = getattr(model, "base_model_prefix", None)
    authenticated_prefix = (
        prefix if isinstance(prefix, str) and prefix.isidentifier() else None
    )
    authoritative_targets = _authoritative_checkpoint_key_targets(
        authenticated_keys,
        live_state=live_state,
        model=model,
    )
    non_executing_keys = _qwen3_5_non_executing_checkpoint_keys(
        authenticated_keys,
        live_state=live_state,
        model=model,
    )
    bindings = _bind_authenticated_live_tensors(
        authenticated_keys,
        live_state=live_state,
        model=model,
        prefix=authenticated_prefix,
        authoritative_targets=authoritative_targets,
    )
    authenticated_names, authenticated_tensors = _verify_authenticated_safetensors(
        checkpoint,
        authenticated_keys=authenticated_keys,
        bindings=bindings,
        model=model,
        prefix=authenticated_prefix,
        safe_open=safe_open,
        non_executing_keys=non_executing_keys,
        authenticated_config=authenticated_config,
    )
    _require_complete_live_state(
        live_state,
        authenticated_names=authenticated_names,
        authenticated_tensors=authenticated_tensors,
    )


def _require_model_config_match(checkpoint: Path, *, model: object) -> object:
    """Compare the live model config with a fresh offline checkpoint load."""

    transformers = importlib.import_module("transformers")
    auto_config = getattr(transformers, "AutoConfig", None)
    from_pretrained = getattr(auto_config, "from_pretrained", None)
    live_config = getattr(model, "config", None)
    live_to_dict = getattr(live_config, "to_dict", None)
    if not callable(from_pretrained) or not callable(live_to_dict):
        raise RuntimeError("strict HF model configuration is unavailable")
    try:
        authenticated_config = from_pretrained(
            str(checkpoint),
            local_files_only=True,
            trust_remote_code=False,
        )
        authenticated_to_dict = getattr(authenticated_config, "to_dict", None)
        if not callable(authenticated_to_dict):
            raise TypeError
        live_payload = live_to_dict()
        authenticated_payload = authenticated_to_dict()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "strict HF checkpoint configuration could not be authenticated"
        ) from exc
    if not isinstance(live_payload, Mapping) or not isinstance(
        authenticated_payload, Mapping
    ):
        raise RuntimeError("strict HF model configuration is unavailable")
    if (
        live_config.__class__.__module__,
        live_config.__class__.__qualname__,
    ) != (
        authenticated_config.__class__.__module__,
        authenticated_config.__class__.__qualname__,
    ):
        raise ValueError(
            "strict HF live model config class does not match the checkpoint"
        )
    normalized_live = cast(dict[str, object], _json_safe_tokenizer_value(live_payload))
    normalized_authenticated = cast(
        dict[str, object], _json_safe_tokenizer_value(authenticated_payload)
    )
    # This locator is not behavioral and differs when an already-created model is
    # saved before being loaded through ``from_pretrained``.
    normalized_live.pop("_name_or_path", None)
    normalized_authenticated.pop("_name_or_path", None)

    # Transformers records dtypes inferred from authenticated safetensor weights
    # on live top-level and component configs even when the authored checkpoint
    # leaves them unspecified. Tensor inventories and values are independently
    # rebound below, so these same-path live-only values are not authored config.
    def drop_inferred_dtypes(
        live: dict[str, object], authenticated: dict[str, object]
    ) -> None:
        for key in tuple(authenticated):
            authenticated_value = authenticated[key]
            if key == "dtype" and authenticated_value is None:
                live.pop(key, None)
                authenticated.pop(key)
                continue
            live_value = live.get(key)
            if isinstance(live_value, dict) and isinstance(authenticated_value, dict):
                drop_inferred_dtypes(live_value, authenticated_value)

    drop_inferred_dtypes(normalized_live, normalized_authenticated)

    # Fine-grained FP8 loaders replace the authored mapping with a typed runtime
    # mapping. Normalize only the exact 5.14.1 defaults and the observed legacy
    # ``fmt=e4m3`` marker. Unknown fields remain a hard mismatch instead of being
    # silently discarded by the runtime's permissive ``**kwargs`` constructor.
    live_quantization = normalized_live.get("quantization_config")
    authenticated_quantization = normalized_authenticated.get("quantization_config")
    if isinstance(live_quantization, dict) and isinstance(
        authenticated_quantization, dict
    ):
        methods = {
            live_quantization.get("quant_method"),
            authenticated_quantization.get("quant_method"),
        }
        if "fp8" in methods:
            if methods != {"fp8"}:
                raise ValueError(
                    "strict HF live model quantization config class does not "
                    "match the checkpoint"
                )
            allowed = {
                "activation_scheme",
                "dequantize",
                "fmt",
                "modules_to_not_convert",
                "quant_method",
                "scale_fmt",
                "weight_block_size",
            }
            if (set(live_quantization) | set(authenticated_quantization)) - allowed:
                raise ValueError(
                    "strict HF fine-grained FP8 config contains unsupported fields"
                )

            if "fmt" in live_quantization:
                raise ValueError(
                    "strict HF live fine-grained FP8 config contains an "
                    "unsupported legacy field"
                )

            def normalize_fp8(
                payload: dict[str, object], *, authenticated: bool
            ) -> dict[str, object]:
                normalized = dict(payload)
                has_legacy_format = "fmt" in normalized
                legacy_format = normalized.pop("fmt", None)
                if has_legacy_format and (not authenticated or legacy_format != "e4m3"):
                    raise ValueError(
                        "strict HF fine-grained FP8 legacy format is unsupported"
                    )
                normalized.setdefault("activation_scheme", "dynamic")
                normalized.setdefault("dequantize", False)
                normalized.setdefault("modules_to_not_convert", None)
                normalized.setdefault("quant_method", "fp8")
                normalized.setdefault("scale_fmt", "float")
                normalized.setdefault("weight_block_size", [128, 128])
                return normalized

            normalized_live["quantization_config"] = normalize_fp8(
                live_quantization, authenticated=False
            )
            normalized_authenticated["quantization_config"] = normalize_fp8(
                authenticated_quantization, authenticated=True
            )
    if normalized_live != normalized_authenticated:
        raise ValueError(
            "strict HF live model config does not match the authenticated checkpoint"
        )
    return authenticated_config


def _require_model_eval_mode(model: object) -> None:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise RuntimeError("strict HF native model does not expose module state")
    try:
        bound_modules = tuple(modules())
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "strict HF native model module state is unavailable"
        ) from exc
    if not bound_modules or any(
        getattr(module, "training", None) is not False for module in bound_modules
    ):
        raise RuntimeError(
            "strict HF causal scoring requires model.eval() for every submodule"
        )


def require_loaded_hf_checkpoint_binding(
    *,
    spec: ModelRuntimeSpec,
    identity: HFSnapshotArtifactIdentity,
    model: object,
    tokenizer: object,
    checkpoint: Path | None = None,
) -> None:
    """Bind the live model/tokenizer to one locally authenticated checkpoint."""

    checkpoint = (
        Path(spec.model_id).expanduser() if checkpoint is None else Path(checkpoint)
    )
    if not checkpoint.is_dir():
        raise RuntimeError(
            "strict HF runtime-behavior evidence requires a materialized local "
            "checkpoint; a remote revision alone cannot bind the loaded model"
        )
    declared_tree = identity.checkpoint_tree_sha256
    if declared_tree is None:
        raise RuntimeError(
            "strict HF runtime-behavior evidence requires checkpoint_tree_sha256"
        )
    try:
        before = checkpoint_tree_sha256(checkpoint).removeprefix("sha256:")
    except (CheckpointIdentityError, OSError) as exc:
        raise RuntimeError(
            "strict HF checkpoint tree could not be authenticated"
        ) from exc
    if before != declared_tree:
        raise ValueError(
            "strict HF checkpoint tree does not match the authenticated identity"
        )
    tokenizer_sha256 = hf_tokenizer_contract_sha256(tokenizer)
    if tokenizer_sha256 != identity.tokenizer_metadata_sha256:
        raise ValueError(
            "strict HF live tokenizer does not match tokenizer_metadata_sha256"
        )
    _require_model_eval_mode(model)
    authenticated_config = _require_model_config_match(checkpoint, model=model)
    _require_safetensors_match(
        checkpoint,
        model=model,
        authenticated_config=authenticated_config,
    )
    try:
        after = checkpoint_tree_sha256(checkpoint).removeprefix("sha256:")
    except (CheckpointIdentityError, OSError) as exc:
        raise RuntimeError(
            "strict HF checkpoint tree could not be reauthenticated"
        ) from exc
    if after != before:
        raise RuntimeError("strict HF checkpoint tree changed during model binding")


@dataclass(frozen=True)
class HFTransformersCausalScorer:
    """Provider-owned deterministic scorer bound to one model and tokenizer."""

    model: object = field(repr=False, compare=False)
    tokenizer: object = field(repr=False, compare=False)
    artifact_identity_sha256: str
    checkpoint_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.artifact_identity_sha256) is None:
            raise ValueError("artifact_identity_sha256 must be a sha256 digest")
        if self.checkpoint_path is not None:
            checkpoint = Path(self.checkpoint_path)
            if not checkpoint.is_absolute():
                raise ValueError("checkpoint_path must be absolute")
            object.__setattr__(self, "checkpoint_path", checkpoint)

    def require_binding(self, *, model: object, artifact_identity_sha256: str) -> None:
        if self.model is not model:
            raise ValueError("strict HF scorer is not bound to the exact native model")
        if self.artifact_identity_sha256 != artifact_identity_sha256:
            raise ValueError(
                "strict HF scorer artifact identity does not match the model spec"
            )

    def __call__(
        self, batch: EvaluationBatch, settings: RuntimeExecutionSettings
    ) -> ScoringObservation:
        if settings.batch_size != 1:
            raise ValueError("strict HF causal scoring currently requires batch_size=1")
        torch = importlib.import_module("torch")
        tokenizer_call = self.tokenizer if callable(self.tokenizer) else None
        decode = getattr(self.tokenizer, "decode", None)
        if (
            not callable(self.model)
            or not callable(tokenizer_call)
            or not callable(decode)
        ):
            raise RuntimeError(
                "strict HF causal scorer requires model and tokenizer APIs"
            )
        _require_model_eval_mode(self.model)

        parameters = tuple(cast(Any, self.model).parameters())
        tensors = parameters or tuple(cast(Any, self.model).buffers())
        if not tensors:
            raise RuntimeError("strict HF native model has no execution tensors")
        device = tensors[0].device
        records = []
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
                    deadline = time.monotonic() + settings.timeout_seconds
                    if batch.task != "text_causal":
                        raise ValueError(
                            "built-in HF execution supports only text_causal"
                        )
                    if record.input_parts:
                        if len(record.input_parts) != 1 or (
                            record.input_parts[0].kind != "text"
                            or record.input_parts[0].role != "prompt"
                        ):
                            raise ValueError(
                                "built-in HF causal execution requires one prompt "
                                "text input part"
                            )
                        expected_input_sha256 = evaluation_input_parts_sha256(
                            record.input_parts
                        )
                    else:
                        expected_input_sha256 = hashlib.sha256(
                            record.input_text.encode("utf-8")
                        ).hexdigest()
                    if expected_input_sha256 != record.input_sha256:
                        raise ValueError(
                            "runtime evaluation input does not match input_sha256"
                        )
                    encoded = tokenizer_call(
                        record.input_text,
                        add_special_tokens=True,
                        max_length=settings.context_length,
                        return_tensors="pt",
                        truncation=True,
                    )
                    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                        raise RuntimeError(
                            "strict HF tokenizer did not return input_ids"
                        )
                    model_inputs = {
                        key: value.to(device)
                        for key, value in encoded.items()
                        if hasattr(value, "to")
                    }
                    input_ids = model_inputs.get("input_ids")
                    if (
                        input_ids is None
                        or input_ids.ndim != 2
                        or input_ids.shape[0] != 1
                    ):
                        raise RuntimeError(
                            "strict HF tokenizer returned invalid input_ids"
                        )
                    if input_ids.shape[1] < 1:
                        raise RuntimeError(
                            "strict HF tokenizer returned empty input_ids"
                        )
                    nll_facts: tuple[float, int, int] | None = None
                    if batch.metric == "normalized_nll_per_utf8_byte":
                        if record.expected_output is None:
                            raise ValueError(
                                "strict HF normalized NLL requires expected_output"
                            )
                        target = tokenizer_call(
                            record.expected_output,
                            add_special_tokens=False,
                            return_tensors="pt",
                            truncation=False,
                        )
                        if not isinstance(target, Mapping) or "input_ids" not in target:
                            raise RuntimeError(
                                "strict HF tokenizer did not return target input_ids"
                            )
                        target_ids = target["input_ids"]
                        if hasattr(target_ids, "to"):
                            target_ids = target_ids.to(device)
                        if (
                            not hasattr(target_ids, "ndim")
                            or target_ids.ndim != 2
                            or target_ids.shape[0] != 1
                            or target_ids.shape[1] < 1
                        ):
                            raise RuntimeError(
                                "strict HF tokenizer returned invalid target input_ids"
                            )
                        target_count = int(target_ids.shape[1])
                        if target_count > settings.max_output_tokens:
                            raise ValueError(
                                "strict HF normalized NLL target exceeds "
                                "max_output_tokens"
                            )
                        prompt_token_ids = [
                            int(input_ids[0, index].item())
                            for index in range(int(input_ids.shape[1]))
                        ]
                        target_token_ids = [
                            int(target_ids[0, index].item())
                            for index in range(target_count)
                        ]
                        decoded_prompt = decode(
                            prompt_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        decoded_continuation = decode(
                            prompt_token_ids + target_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        if (
                            not isinstance(decoded_prompt, str)
                            or not isinstance(decoded_continuation, str)
                            or decoded_continuation
                            != decoded_prompt + record.expected_output
                        ):
                            raise ValueError(
                                "strict HF normalized NLL target is not an exact "
                                "tokenizer continuation of the prompt"
                            )
                        history = input_ids
                        target_logprobs: list[float] = []
                        for target_index in range(target_count):
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "strict HF normalized NLL scoring timed out"
                                )
                            window = history[:, -settings.context_length :]
                            attention_mask = torch.ones_like(window)
                            result = self.model(
                                input_ids=window,
                                attention_mask=attention_mask,
                                return_dict=True,
                                use_cache=False,
                            )
                            logits = getattr(result, "logits", None)
                            if (
                                logits is None
                                or logits.ndim != 3
                                or logits.shape[0] != 1
                                or logits.shape[1] != window.shape[1]
                                or not bool(torch.isfinite(logits[:, -1, :]).all())
                            ):
                                raise RuntimeError(
                                    "strict HF model returned invalid causal logits"
                                )
                            target_token = target_ids[
                                :, target_index : target_index + 1
                            ]
                            if (
                                int(target_token.item()) < 0
                                or int(target_token.item()) >= logits.shape[-1]
                            ):
                                raise RuntimeError(
                                    "strict HF target token is outside the model vocabulary"
                                )
                            token_logprob = torch.log_softmax(
                                logits[:, -1, :], dim=-1
                            ).gather(1, target_token)
                            value = float(token_logprob.item())
                            if not math.isfinite(value) or value > 0:
                                raise RuntimeError(
                                    "strict HF model returned an invalid target logprob"
                                )
                            target_logprobs.append(value)
                            history = torch.cat((history, target_token), dim=1)
                        nll_facts = (
                            math.fsum(target_logprobs),
                            target_count,
                            len(record.expected_output.encode("utf-8")),
                        )
                    output_text: str | None = None
                    if batch.metric == "exact_match":
                        generated = input_ids
                        new_tokens: list[int] = []
                        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                        for _ in range(settings.max_output_tokens):
                            if time.monotonic() >= deadline:
                                raise TimeoutError("strict HF causal scoring timed out")
                            window = generated[:, -settings.context_length :]
                            attention_mask = torch.ones_like(window)
                            result = self.model(
                                input_ids=window,
                                attention_mask=attention_mask,
                                return_dict=True,
                                use_cache=False,
                            )
                            logits = getattr(result, "logits", None)
                            if (
                                logits is None
                                or logits.ndim != 3
                                or logits.shape[0] != 1
                                or logits.shape[1] != window.shape[1]
                                or not bool(torch.isfinite(logits[:, -1, :]).all())
                            ):
                                raise RuntimeError(
                                    "strict HF model returned invalid causal logits"
                                )
                            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                            token_id = int(next_token.item())
                            new_tokens.append(token_id)
                            generated = torch.cat((generated, next_token), dim=1)
                            if (
                                isinstance(eos_token_id, int)
                                and not isinstance(eos_token_id, bool)
                                and token_id == eos_token_id
                            ):
                                break
                        assert callable(decode)
                        output_text = decode(
                            new_tokens,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=True,
                        )
                        try:
                            output_text = exact_match_output_text(output_text)
                        except ValueError as exc:
                            raise RuntimeError(
                                "strict HF tokenizer returned invalid user-visible text"
                            ) from exc
                    records.append(
                        RuntimeScoringRecord(
                            record_id=record.record_id,
                            input_sha256=record.input_sha256,
                            status="ok",
                            output_text=output_text,
                            output_sha256=(
                                _sha256(output_text.encode("utf-8"))
                                if output_text is not None
                                else None
                            ),
                            logprob_sum=(nll_facts[0] if nll_facts else None),
                            token_count=(nll_facts[1] if nll_facts else None),
                            utf8_byte_count=(nll_facts[2] if nll_facts else None),
                        )
                    )
        finally:
            torch.use_deterministic_algorithms(
                deterministic_enabled,
                warn_only=deterministic_warn_only,
            )
        aggregate_source_sha256 = runtime_scoring_records_sha256(
            [cast(dict[str, object], asdict(record)) for record in records]
        )
        return ScoringObservation(
            provider_name=HFTransformersProvider.name,
            artifact_identity_sha256=self.artifact_identity_sha256,
            schedule_sha256=batch.schedule_sha256,
            records=tuple(records),
            aggregate_source_sha256=aggregate_source_sha256,
        )


def _require_strict_execution_binding(
    *,
    spec: ModelRuntimeSpec,
    identity: HFSnapshotArtifactIdentity,
    context: RuntimeExecutionContext,
) -> RuntimeScorer:
    scorer = context.scorer
    if not isinstance(scorer, HFTransformersCausalScorer):
        raise RuntimeError(
            "strict HF runtime-behavior evidence requires the provider-owned "
            "HFTransformersCausalScorer; arbitrary scorer callbacks are not "
            "authenticated"
        )
    assert context.provider_state is not None
    expected_identity_sha256 = artifact_identity_sha256(identity)
    scorer.require_binding(
        model=context.provider_state,
        artifact_identity_sha256=expected_identity_sha256,
    )
    require_loaded_hf_checkpoint_binding(
        spec=spec,
        identity=identity,
        model=context.provider_state,
        tokenizer=scorer.tokenizer,
        checkpoint=scorer.checkpoint_path,
    )
    return scorer


@dataclass(frozen=True)
class _HFReceiptProvenance:
    plugin: RuntimeProviderPluginIdentity
    backend: RuntimeBackendIdentity
    capabilities: RuntimeProviderCapabilities
    artifact_identity: HFSnapshotArtifactIdentity
    execution_settings: RuntimeExecutionSettings
    device: RuntimeDeviceFacts
    outer_image_digest: str


@dataclass
class _HFTransformersSession:
    _scorer: RuntimeScorer
    _close_callback: Callable[[], None] | None = None
    _artifact_identity_sha256: str | None = None
    _receipt_provenance: _HFReceiptProvenance | None = None
    _revalidate_binding: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    _latest_observation_sha256: str | None = None
    _closed: bool = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime provider session is closed")

    def score(self, batch: EvaluationBatch) -> ScoringObservation:
        """Delegate scoring and fail closed on schedule/pairing drift."""

        self._require_open()
        self._latest_observation_sha256 = None
        provenance = self._receipt_provenance
        if provenance is None:
            direct_scorer = cast(
                Callable[[EvaluationBatch], ScoringObservation], self._scorer
            )
            observation = direct_scorer(batch)
        else:
            if self._revalidate_binding is not None:
                self._revalidate_binding()
            try:
                observation = self._scorer(batch, provenance.execution_settings)
            except TypeError as exc:
                raise RuntimeError(
                    "strict hf_transformers scorer must accept the exact runtime "
                    "execution settings contract"
                ) from exc
            finally:
                if self._revalidate_binding is not None:
                    self._revalidate_binding()
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
        from invarlock.runtime_provider_evidence import encode_scoring_observation

        self._latest_observation_sha256 = _sha256(
            encode_scoring_observation(observation)
        )
        return observation

    def runtime_receipt(self) -> RuntimeProviderReceipt:
        """Return strict provenance bound to the latest complete observation."""

        self._require_open()
        if self._receipt_provenance is None:
            raise RuntimeError(
                "runtime provider receipt requires strict authenticated execution"
            )
        if self._latest_observation_sha256 is None:
            raise RuntimeError("runtime provider receipt is unavailable before scoring")
        provenance = self._receipt_provenance
        return RuntimeProviderReceipt(
            plugin=provenance.plugin,
            backend=provenance.backend,
            capabilities=provenance.capabilities,
            artifact_identity=provenance.artifact_identity,
            execution_settings=provenance.execution_settings,
            device=provenance.device,
            outer_image_digest=provenance.outer_image_digest,
            scoring_observation_sha256=self._latest_observation_sha256,
        )

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
            metrics=(
                "exact_match",
                "normalized_nll_per_utf8_byte",
            ),
            execution_modes=("in_process",),
            required_extra="hf",
            required_image=None,
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

    def authenticate_artifact(
        self, spec: ModelRuntimeSpec, artifact_path: Path
    ) -> HFSnapshotArtifactIdentity:
        """Authenticate local checkpoint bytes without importing model runtimes."""

        identity = self.identify_artifact(spec)
        expected_tree = identity.checkpoint_tree_sha256
        if expected_tree is None:
            raise ValueError(
                "hf_transformers authentication requires checkpoint_tree_sha256"
            )
        try:
            observed_tree = checkpoint_tree_sha256(artifact_path).removeprefix(
                "sha256:"
            )
        except (CheckpointIdentityError, OSError) as exc:
            raise ValueError(
                "hf_transformers checkpoint could not be authenticated"
            ) from exc
        if observed_tree != expected_tree:
            raise ValueError("hf_transformers checkpoint tree digest does not match")
        return identity

    def prepare_execution(
        self,
        spec: ModelRuntimeSpec,
        resources: RuntimeArtifactResources,
    ) -> RuntimeExecutionContext:
        """Load and authenticate one local checkpoint without network or remote code."""

        self.validate_config(spec)
        resources.require_support_names(frozenset())
        checkpoint = resources.primary_path()
        if not checkpoint.is_dir():
            raise ValueError("hf_transformers primary artifact must be a directory")
        identity = self.authenticate_artifact(spec, checkpoint)

        transformers = importlib.import_module("transformers")
        auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
        tokenizer_from_pretrained = getattr(auto_tokenizer, "from_pretrained", None)
        auto_model = getattr(transformers, "AutoModelForCausalLM", None)
        model_from_pretrained = getattr(auto_model, "from_pretrained", None)
        if not callable(tokenizer_from_pretrained):
            raise RuntimeError("transformers AutoTokenizer is unavailable")
        if not callable(model_from_pretrained):
            raise RuntimeError("transformers AutoModelForCausalLM is unavailable")
        tokenizer = tokenizer_from_pretrained(
            str(checkpoint),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = load_hf_model_with_strict_loading_info(
            model_from_pretrained,
            checkpoint,
        )
        move_to_device = getattr(model, "to", None)
        if not callable(move_to_device):
            raise RuntimeError("loaded HF model does not expose to()")
        model = move_to_device(resources.device_kind)
        evaluate = getattr(model, "eval", None)
        if not callable(evaluate):
            raise RuntimeError("loaded HF model does not expose eval()")
        evaluate()
        identity_sha256 = artifact_identity_sha256(identity)
        scorer = HFTransformersCausalScorer(
            model=model,
            tokenizer=tokenizer,
            artifact_identity_sha256=identity_sha256,
            checkpoint_path=checkpoint,
        )
        require_loaded_hf_checkpoint_binding(
            spec=spec,
            identity=identity,
            model=model,
            tokenizer=tokenizer,
            checkpoint=checkpoint,
        )
        return RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=resources.container_image_digest,
            device_kind=resources.device_kind,
            artifact_identity_sha256=identity_sha256,
            provider_state=model,
            scorer=scorer,
        )

    def open(
        self,
        spec: ModelRuntimeSpec,
        context: RuntimeExecutionContext,
    ) -> _HFTransformersSession:
        self.validate_config(spec)
        if context.strict:
            _require_strict_runtime_boundary(context)
        if context.strict and context.artifact_identity_sha256 is None:
            raise ValueError(
                "strict hf_transformers execution requires artifact_identity_sha256"
            )
        for field_name in ("provider_state", "scorer"):
            if getattr(context, field_name) is None:
                raise ValueError(
                    f"hf_transformers requires prebound {field_name} in context"
                )
        identity = self.identify_artifact(spec)
        if context.artifact_identity_sha256 is not None:
            expected_identity_sha256 = artifact_identity_sha256(identity)
            if context.artifact_identity_sha256 != expected_identity_sha256:
                raise ValueError(
                    "runtime context artifact identity does not match model spec"
                )
        provenance: _HFReceiptProvenance | None = None
        revalidate_binding: Callable[[], None] | None = None
        runtime_scorer = cast(RuntimeScorer, context.scorer)
        if context.strict:
            runtime_scorer = _require_strict_execution_binding(
                spec=spec,
                identity=identity,
                context=context,
            )
            image_digest = context.container_image_digest
            assert image_digest is not None
            execution_settings = _strict_execution_settings(spec)
            backend = _installed_backend_identity(context.provider_state)
            device = _observed_device_facts(
                context.provider_state,
                expected_device_kind=context.device_kind,
            )
            provenance = _HFReceiptProvenance(
                plugin=RuntimeProviderPluginIdentity(
                    name=self.name,
                    distribution="invarlock",
                    distribution_version=INVARLOCK_VERSION,
                ),
                backend=backend,
                capabilities=self.capabilities(),
                artifact_identity=identity,
                execution_settings=execution_settings,
                device=device,
                outer_image_digest=image_digest,
            )
            if isinstance(runtime_scorer, HFTransformersCausalScorer):
                model = context.provider_state
                tokenizer = runtime_scorer.tokenizer

                def revalidate_binding() -> None:
                    require_loaded_hf_checkpoint_binding(
                        spec=spec,
                        identity=identity,
                        model=model,
                        tokenizer=tokenizer,
                        checkpoint=runtime_scorer.checkpoint_path,
                    )

        return _HFTransformersSession(
            _scorer=runtime_scorer,
            _close_callback=context.close_callback,
            _artifact_identity_sha256=context.artifact_identity_sha256,
            _receipt_provenance=provenance,
            _revalidate_binding=revalidate_binding,
        )


__all__ = [
    "HFTransformersCausalScorer",
    "HFTransformersProvider",
    "INVARLOCK_RUNTIME_PROVIDER_ABI",
    "hf_tokenizer_contract_sha256",
    "load_hf_model_with_strict_loading_info",
    "require_loaded_hf_checkpoint_binding",
]
