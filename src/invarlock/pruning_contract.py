"""Package-owned, fail-closed contract for magnitude-pruning evidence.

The pruning materializer, raw-artifact replay, clean-selection verifier, and
evidence-pack verifier must all use the same architecture and storage policy.
An unknown model family, noncanonical target manifest, or unsupported storage
representation is therefore an unsupported subject rather than a best-effort
edit.  Versioned schemas intentionally retire evidence produced under the
earlier split policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from .evidence_pack_json import StrictJsonError, load_json_object

PRUNING_SCOPE_POLICY_VERSION = "architecture-aware-pruning-v1"
PRUNING_REPLAY_SCHEMA = "invarlock/magnitude-prune-replay-v1"
PRUNING_ALGORITHM = "exact_magnitude_flattened_ties_v1"
PRUNING_STORAGE_POLICY = "regular-unquantized-safetensors-v1"
PRUNING_TARGET_MANIFEST_SCHEMA = "invarlock/pruning-target-manifest-v1"
PRUNING_SCOPES = frozenset({"ffn", "attn", "all"})
PRUNING_SUPPORTED_FLOAT_DTYPES = frozenset(
    {"torch.float16", "torch.float32", "torch.float64", "torch.bfloat16"}
)


class PruningContractError(ValueError):
    """Raised when a checkpoint or proof is outside the pruning contract."""


@dataclass(frozen=True)
class PruningCheckpointContract:
    """Architecture and configuration identity used for target selection."""

    model_type: str
    architecture: str
    config_sha256: str


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MODEL_TYPE_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_TENSOR_NAME_RE = re.compile(r"[^\x00\r\n]+\Z")
_EXCLUDED_PATH_SEGMENTS = frozenset(
    {
        "audio",
        "connector",
        "image",
        "images",
        "mm_projector",
        "multi_modal_projector",
        "multi_token_prediction",
        "multi_token_predictor",
        "multimodal",
        "mtp",
        "vision",
        "vision_encoder",
        "vision_model",
        "vision_resampler",
        "vision_tower",
        "visual",
    }
)

_DECODER_MODEL_TYPES = frozenset(
    {
        "gemma",
        "gemma2",
        "llama",
        "mistral",
        "olmo",
        "olmo2",
        "qwen2",
        "qwen3",
    }
)

MODEL_TYPE_ARCHITECTURES: Mapping[str, str] = MappingProxyType(
    {
        **dict.fromkeys(_DECODER_MODEL_TYPES, "decoder"),
        "falcon": "falcon",
        "gpt2": "gpt2",
        "gpt_bigcode": "gpt2",
        "gpt_neox": "gpt_neox",
        "bloom": "gpt_neox",
        "opt": "opt",
        "bert": "bert",
        "roberta": "bert",
        "distilbert": "distilbert",
    }
)
# This is deliberately narrower than the loader/adapter support matrix.  A
# raw transformation or pruning ``all`` scope must enumerate every reviewed
# language-model attention/FFN/router matrix, not merely a familiar subset.
# Families with fused, linear, recurrent, or evolving MoE layouts remain
# unsupported until an authentic checkpoint-key corpus and explicit policy
# prove complete coverage across generator and verifier boundaries.
SUPPORTED_PRUNING_ARCHITECTURES = frozenset(MODEL_TYPE_ARCHITECTURES.values())

_DECODER_FFN = re.compile(
    r"(?:^|\.)mlp\.(?:(?:shared_)?expert\.)?(?:gate_proj|up_proj|down_proj|fc1|fc2|w1|w2|w3)\.weight$"
)
_DECODER_EXPERT = re.compile(
    r"(?:^|\.)(?:mlp\.)?experts\.\d+\.(?:gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$"
)
_DECODER_ATTN = re.compile(
    r"(?:^|\.)(?:self_attn|attention)\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$"
)
_DECODER_ROUTER = re.compile(
    r"(?:^|\.)(?:mlp|block_sparse_moe)\.(?:gate|router)\.weight$"
)
_MIXTRAL_FFN = re.compile(
    r"(?:^|\.)block_sparse_moe\.experts\.\d+\.(?:w1|w2|w3)\.weight$"
)
_MIXTRAL_ROUTER = re.compile(r"(?:^|\.)block_sparse_moe\.(?:gate|router)\.weight$")
_FALCON_FFN = re.compile(r"(?:^|\.)mlp\.(?:dense_h_to_4h|dense_4h_to_h)\.weight$")
_FALCON_ATTN = re.compile(
    r"(?:^|\.)(?:self_attention|attention)\.(?:query_key_value|dense|q_proj|k_proj|v_proj|o_proj)\.weight$"
)
_GPT2_FFN = re.compile(r"(?:^|\.)mlp\.(?:c_fc|c_proj)\.weight$")
_GPT2_ATTN = re.compile(r"(?:^|\.)attn\.(?:c_attn|c_proj)\.weight$")
_GPT_NEOX_FFN = re.compile(r"(?:^|\.)mlp\.(?:dense_h_to_4h|dense_4h_to_h)\.weight$")
_GPT_NEOX_ATTN = re.compile(
    r"(?:^|\.)(?:attention|self_attention)\.(?:query_key_value|dense)\.weight$"
)
_OPT_FFN = re.compile(r"(?:^|\.)(?:fc1|fc2)\.weight$")
_OPT_ATTN = re.compile(r"(?:^|\.)self_attn\.(?:q_proj|k_proj|v_proj|out_proj)\.weight$")
_BERT_FFN = re.compile(r"(?:^|\.)(?:intermediate|output)\.dense\.weight$")
_BERT_ATTN = re.compile(
    r"(?:^|\.)attention\.(?:self\.(?:query|key|value)|output\.dense)\.weight$"
)
_DISTILBERT_FFN = re.compile(r"(?:^|\.)ffn\.(?:lin1|lin2)\.weight$")
_DISTILBERT_ATTN = re.compile(
    r"(?:^|\.)attention\.(?:q_lin|k_lin|v_lin|out_lin)\.weight$"
)

_TARGET_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "scope",
        "scope_policy",
        "pruning_algorithm",
        "storage_policy",
        "model_type",
        "architecture",
        "config_sha256",
        "targets",
    }
)
_TARGET_FIELDS = frozenset({"name", "dtype", "shape", "numel"})


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _path_segments(name: str) -> tuple[str, ...]:
    return tuple(
        segment for segment in re.split(r"[^a-z0-9_]+", name.lower()) if segment
    )


def _is_excluded_path(name: str) -> bool:
    for segment in _path_segments(name):
        if segment in _EXCLUDED_PATH_SEGMENTS:
            return True
        if segment == "aux" or segment.startswith(("aux_", "auxiliary")):
            return True
    return False


def _normalize_model_type(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PruningContractError(f"{label} must declare model_type")
    model_type = value.strip().lower().replace("-", "_")
    if _MODEL_TYPE_RE.fullmatch(model_type) is None:
        raise PruningContractError(f"{label} model_type is invalid")
    return model_type


def _validate_storage_policy(config: Mapping[str, Any], model_type: str) -> None:
    if model_type == "gpt_oss":
        raise PruningContractError(
            "magnitude-prune does not support GPT-OSS/MXFP4 storage; use an "
            "unquantized floating-point checkpoint or a backend-specific verifier"
        )
    quantization_config = config.get("quantization_config")
    if quantization_config not in (None, {}):
        raise PruningContractError(
            "magnitude-prune requires an unquantized floating-point checkpoint; "
            "config.json declares quantized storage"
        )
    config_text = json.dumps(config, allow_nan=False, sort_keys=True).lower()
    if "mxfp4" in config_text:
        raise PruningContractError(
            "magnitude-prune does not support MXFP4 storage; use an unquantized "
            "floating-point checkpoint or a backend-specific verifier"
        )


def _validate_contract_identity(contract: PruningCheckpointContract) -> None:
    model_type = _normalize_model_type(
        contract.model_type, label="pruning checkpoint contract"
    )
    if model_type != contract.model_type:
        raise PruningContractError(
            "pruning checkpoint contract model_type is not canonical"
        )
    if MODEL_TYPE_ARCHITECTURES.get(model_type) != contract.architecture:
        raise PruningContractError(
            "pruning checkpoint contract model_type and architecture mismatch"
        )
    if _SHA256_RE.fullmatch(contract.config_sha256) is None:
        raise PruningContractError(
            "pruning checkpoint contract config_sha256 is invalid"
        )


def checkpoint_pruning_contract(checkpoint_dir: Path) -> PruningCheckpointContract:
    """Resolve the only supported target-selection policy for a checkpoint."""

    config_path = checkpoint_dir / "config.json"
    try:
        config = load_json_object(config_path, label="checkpoint config.json")
    except StrictJsonError as exc:
        raise PruningContractError(f"checkpoint config.json is invalid: {exc}") from exc
    model_type = _normalize_model_type(
        config.get("model_type"), label="checkpoint config.json"
    )
    _validate_storage_policy(config, model_type)
    architecture = MODEL_TYPE_ARCHITECTURES.get(model_type)
    if architecture is None:
        raise PruningContractError(
            "magnitude-prune has no explicit resolver for model_type="
            f"{model_type!r}; add a model-family-specific transformation and replay "
            "contract before using this checkpoint"
        )
    return PruningCheckpointContract(
        model_type=model_type,
        architecture=architecture,
        config_sha256=_canonical_json_sha256(config),
    )


def validate_pruning_scope(scope: str) -> str:
    if not isinstance(scope, str):
        raise PruningContractError(
            f"magnitude-prune scope must be one of {sorted(PRUNING_SCOPES)}"
        )
    normalized = scope.strip().lower()
    if normalized not in PRUNING_SCOPES:
        raise PruningContractError(
            f"magnitude-prune scope must be one of {sorted(PRUNING_SCOPES)}"
        )
    return normalized


def pruning_target_role(
    name: str,
    *,
    contract: PruningCheckpointContract,
    ndim: int,
) -> str | None:
    """Return ``ffn``, ``attn``, or ``router`` for one supported matrix."""

    _validate_contract_identity(contract)
    if ndim < 2 or _is_excluded_path(name):
        return None
    architecture = contract.architecture
    if architecture == "decoder":
        if _DECODER_FFN.search(name) or _DECODER_EXPERT.search(name):
            return "ffn"
        if _DECODER_ATTN.search(name):
            return "attn"
        if _DECODER_ROUTER.search(name):
            return "router"
        return None
    if architecture == "falcon":
        if _FALCON_FFN.search(name):
            return "ffn"
        if _FALCON_ATTN.search(name):
            return "attn"
        return None
    if architecture == "gpt2":
        if _GPT2_FFN.search(name):
            return "ffn"
        if _GPT2_ATTN.search(name):
            return "attn"
        return None
    if architecture == "gpt_neox":
        if _GPT_NEOX_FFN.search(name):
            return "ffn"
        if _GPT_NEOX_ATTN.search(name):
            return "attn"
        return None
    if architecture == "opt":
        if _OPT_FFN.search(name):
            return "ffn"
        if _OPT_ATTN.search(name):
            return "attn"
        return None
    if architecture == "bert":
        if _BERT_FFN.search(name):
            return "ffn"
        if _BERT_ATTN.search(name):
            return "attn"
        return None
    if architecture == "distilbert":
        if _DISTILBERT_FFN.search(name):
            return "ffn"
        if _DISTILBERT_ATTN.search(name):
            return "attn"
        return None
    raise AssertionError(f"unknown pruning architecture: {architecture}")


def is_pruning_target(
    name: str,
    *,
    scope: str,
    contract: PruningCheckpointContract,
    ndim: int,
) -> bool:
    """Determine membership under the versioned architecture policy.

    ``all`` is exactly FFN, attention, and MoE-router matrices.  It excludes
    embeddings, heads, norms, vision, audio, connector, and MTP paths.
    """

    normalized_scope = validate_pruning_scope(scope)
    role = pruning_target_role(name, contract=contract, ndim=ndim)
    if normalized_scope == "all":
        return role in {"ffn", "attn", "router"}
    return role == normalized_scope


def pruning_target_entry(name: str, tensor: Any) -> dict[str, object]:
    """Return a target descriptor without importing a tensor backend."""

    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": [int(dimension) for dimension in tensor.shape],
        "numel": int(tensor.numel()),
    }


def _canonical_target_descriptor(target: object) -> dict[str, object]:
    if not isinstance(target, Mapping) or set(target) != _TARGET_FIELDS:
        raise PruningContractError(
            "pruning target descriptor must contain only name, dtype, shape, and numel"
        )
    name = target["name"]
    dtype = target["dtype"]
    shape = target["shape"]
    numel = target["numel"]
    if not isinstance(name, str) or _TENSOR_NAME_RE.fullmatch(name) is None:
        raise PruningContractError("pruning target name is invalid")
    if not isinstance(dtype, str) or dtype not in PRUNING_SUPPORTED_FLOAT_DTYPES:
        raise PruningContractError(
            "pruning target dtype is not supported floating-point storage"
        )
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        )
    ):
        raise PruningContractError("pruning target shape is invalid")
    if isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0:
        raise PruningContractError("pruning target numel is invalid")
    if numel != math.prod(shape):
        raise PruningContractError("pruning target numel does not match shape")
    return {"name": name, "dtype": dtype, "shape": list(shape), "numel": numel}


def _manifest_contract(manifest: Mapping[str, object]) -> PruningCheckpointContract:
    model_type = _normalize_model_type(
        manifest.get("model_type"), label="pruning target manifest"
    )
    if manifest.get("model_type") != model_type:
        raise PruningContractError(
            "pruning target manifest model_type is not canonical"
        )
    architecture = manifest.get("architecture")
    config_sha256 = manifest.get("config_sha256")
    if not isinstance(architecture, str):
        raise PruningContractError("pruning target manifest architecture is invalid")
    if (
        not isinstance(config_sha256, str)
        or _SHA256_RE.fullmatch(config_sha256) is None
    ):
        raise PruningContractError("pruning target manifest config_sha256 is invalid")
    contract = PruningCheckpointContract(
        model_type=model_type,
        architecture=architecture,
        config_sha256=config_sha256,
    )
    _validate_contract_identity(contract)
    return contract


def validate_pruning_target_manifest(
    manifest: object,
    *,
    expected_scope: str | None = None,
    expected_contract: PruningCheckpointContract | None = None,
) -> dict[str, object]:
    """Validate and return one exact canonical pruning target manifest.

    This is intentionally usable at evidence-pack verification time, where the
    original checkpoint cannot be assumed to be present.  It checks the policy
    binding, supported model family/storage claim, canonical target ordering,
    supported dtypes, and every target's architecture-aware role.
    """

    if not isinstance(manifest, Mapping) or set(manifest) != _TARGET_MANIFEST_FIELDS:
        raise PruningContractError(
            "pruning target manifest has missing or unbound fields"
        )
    if manifest.get("schema") != PRUNING_TARGET_MANIFEST_SCHEMA:
        raise PruningContractError("pruning target manifest schema is invalid")
    if manifest.get("scope_policy") != PRUNING_SCOPE_POLICY_VERSION:
        raise PruningContractError("pruning target manifest scope_policy is invalid")
    if manifest.get("pruning_algorithm") != PRUNING_ALGORITHM:
        raise PruningContractError(
            "pruning target manifest pruning_algorithm is invalid"
        )
    if manifest.get("storage_policy") != PRUNING_STORAGE_POLICY:
        raise PruningContractError("pruning target manifest storage_policy is invalid")
    raw_scope = manifest.get("scope")
    if not isinstance(raw_scope, str):
        raise PruningContractError("pruning target manifest scope is invalid")
    scope = validate_pruning_scope(raw_scope)
    if expected_scope is not None and scope != validate_pruning_scope(expected_scope):
        raise PruningContractError(
            "pruning target manifest scope does not match replay"
        )
    contract = _manifest_contract(manifest)
    if expected_contract is not None and contract != expected_contract:
        raise PruningContractError(
            "pruning target manifest checkpoint identity does not match replay"
        )
    raw_targets = manifest.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise PruningContractError(
            "pruning target manifest must retain selected tensors"
        )
    targets: list[dict[str, object]] = []
    for raw_target in raw_targets:
        target = _canonical_target_descriptor(raw_target)
        name = str(target["name"])
        shape = target["shape"]
        if not isinstance(shape, list):  # narrowed by _canonical_target_descriptor
            raise AssertionError("pruning target shape must be a list")
        if not is_pruning_target(
            name,
            scope=scope,
            contract=contract,
            ndim=len(shape),
        ):
            raise PruningContractError(
                f"pruning target {name!r} is outside the canonical pruning scope"
            )
        targets.append(target)
    names = [str(target["name"]) for target in targets]
    if names != sorted(names) or len(names) != len(set(names)):
        raise PruningContractError(
            "pruning target manifest targets must be sorted and unique"
        )
    canonical: dict[str, object] = {
        "schema": PRUNING_TARGET_MANIFEST_SCHEMA,
        "scope": scope,
        "scope_policy": PRUNING_SCOPE_POLICY_VERSION,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "model_type": contract.model_type,
        "architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "targets": targets,
    }
    if dict(manifest) != canonical:
        raise PruningContractError("pruning target manifest is not canonical")
    return canonical


def pruning_target_manifest(
    *,
    scope: str,
    contract: PruningCheckpointContract,
    targets: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Build a canonical target manifest for exactly selected tensors."""

    _validate_contract_identity(contract)
    normalized_targets = [_canonical_target_descriptor(target) for target in targets]
    normalized_targets.sort(key=lambda target: str(target["name"]))
    return validate_pruning_target_manifest(
        {
            "schema": PRUNING_TARGET_MANIFEST_SCHEMA,
            "scope": validate_pruning_scope(scope),
            "scope_policy": PRUNING_SCOPE_POLICY_VERSION,
            "pruning_algorithm": PRUNING_ALGORITHM,
            "storage_policy": PRUNING_STORAGE_POLICY,
            "model_type": contract.model_type,
            "architecture": contract.architecture,
            "config_sha256": contract.config_sha256,
            "targets": normalized_targets,
        },
        expected_contract=contract,
    )


def pruning_target_manifest_sha256(manifest: Mapping[str, object]) -> str:
    """Return a digest only for an exact canonical target manifest."""

    return _canonical_json_sha256(validate_pruning_target_manifest(manifest))


def finite_pruning_sparsity(value: object) -> float:
    if isinstance(value, bool):
        raise PruningContractError("magnitude-prune sparsity must be a finite number")
    try:
        sparsity = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise PruningContractError(
            "magnitude-prune sparsity must be a finite number"
        ) from exc
    if not math.isfinite(sparsity) or not 0.0 < sparsity < 1.0:
        raise PruningContractError("magnitude-prune sparsity must be in (0, 1)")
    return sparsity


__all__ = [
    "MODEL_TYPE_ARCHITECTURES",
    "PRUNING_ALGORITHM",
    "PRUNING_REPLAY_SCHEMA",
    "PRUNING_SCOPE_POLICY_VERSION",
    "PRUNING_SCOPES",
    "PRUNING_STORAGE_POLICY",
    "PRUNING_SUPPORTED_FLOAT_DTYPES",
    "PRUNING_TARGET_MANIFEST_SCHEMA",
    "SUPPORTED_PRUNING_ARCHITECTURES",
    "PruningCheckpointContract",
    "PruningContractError",
    "checkpoint_pruning_contract",
    "finite_pruning_sparsity",
    "is_pruning_target",
    "pruning_target_entry",
    "pruning_target_manifest",
    "pruning_target_manifest_sha256",
    "pruning_target_role",
    "validate_pruning_scope",
    "validate_pruning_target_manifest",
]
