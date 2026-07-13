"""Fail-closed contract for verifier-grade generated transformations.

This module deliberately describes only generated edit families that can be
replayed from public, canonical parameters.  It does not make an unsupported
simulation acceptable merely because an older generator can emit one.  In
particular, FP8 simulation and dense low-rank approximation require dedicated
storage/replay contracts before they can become verifier-grade lanes.

The target resolver reuses the explicit architecture rules from the pruning
contract.  That avoids the broad substring matching used by legacy generators.
The installed verifier owns a separate target-manifest parser; it must never
trust this generator-facing resolver as its replay oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_json import StrictJsonError, load_json_object
from invarlock.pruning_contract import (
    MODEL_TYPE_ARCHITECTURES,
    PruningCheckpointContract,
    PruningContractError,
    checkpoint_pruning_contract,
    pruning_target_role,
)

TRANSFORMATION_CONTRACT_VERSION = "verifier-grade-transformation-v1"
TRANSFORMATION_SCOPE_POLICY_VERSION = "architecture-aware-transformation-v1"
TRANSFORMATION_PARAMETERS_SCHEMA = "invarlock/transformation-parameters-v1"
TRANSFORMATION_TARGET_MANIFEST_SCHEMA = "invarlock/transformation-target-manifest-v1"

GROUPWISE_RTN_DEQUANTIZED_ALGORITHM = "groupwise_rtn_dequantized_per_row_v1"
SYNTHETIC_LOWRANK_DELTA_ALGORITHM = "deterministic_synthetic_lowrank_delta_v1"
SYNTHETIC_DENSE_UPDATE_ALGORITHM = "deterministic_synthetic_dense_update_v1"
MAX_SYNTHETIC_LOWRANK_RANK = 32
MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS = 16

QUANT_RTN = "quant_rtn"
SYNTHETIC_LOWRANK_DELTA = "synthetic_lowrank_delta"
SYNTHETIC_DENSE_UPDATE = "synthetic_dense_update"
FP8_QUANT = "fp8_quant"
LOWRANK_SVD = "lowrank_svd"

VERIFIER_GRADE_GENERATED_EDIT_TYPES = frozenset(
    {QUANT_RTN, SYNTHETIC_LOWRANK_DELTA, SYNTHETIC_DENSE_UPDATE}
)
UNSUPPORTED_VERIFIER_GRADE_EDIT_TYPES = frozenset({FP8_QUANT, LOWRANK_SVD})
TRANSFORMATION_SCOPES = frozenset({"ffn", "attn", "all"})

_ALGORITHMS = {
    QUANT_RTN: GROUPWISE_RTN_DEQUANTIZED_ALGORITHM,
    SYNTHETIC_LOWRANK_DELTA: SYNTHETIC_LOWRANK_DELTA_ALGORITHM,
    SYNTHETIC_DENSE_UPDATE: SYNTHETIC_DENSE_UPDATE_ALGORITHM,
}
_TARGET_INPUT_KEYS = frozenset({"name", "dtype", "shape", "numel"})
_TARGET_MANIFEST_KEYS = _TARGET_INPUT_KEYS | frozenset({"role", "layer"})
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NONNEGATIVE_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")

_LAYER_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "decoder": (re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),),
    "mixtral": (re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),),
    "falcon": (
        re.compile(r"(?:^|\.)transformer\.h\.(\d+)(?:\.|$)"),
        re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    ),
    "gpt2": (re.compile(r"(?:^|\.)transformer\.h\.(\d+)(?:\.|$)"),),
    "gpt_neox": (
        re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
        re.compile(r"(?:^|\.)transformer\.h\.(\d+)(?:\.|$)"),
    ),
    "opt": (re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),),
    "bert": (re.compile(r"(?:^|\.)layer\.(\d+)(?:\.|$)"),),
    "distilbert": (re.compile(r"(?:^|\.)layer\.(\d+)(?:\.|$)"),),
}

# A raw-transformation target must name one of the layers the checkpoint
# configuration actually declares.  Tensor names are deliberately not used to
# infer this count: an injected ``layers.999`` tensor can look syntactically
# plausible while the model loader silently ignores it.  Keep the model-type
# mapping explicit so a new family cannot inherit an accidental alias.
_LAYER_COUNT_CONFIG_PATHS: dict[str, tuple[str, ...]] = {
    "gemma": ("num_hidden_layers",),
    "gemma2": ("num_hidden_layers",),
    "llama": ("num_hidden_layers",),
    "mistral": ("num_hidden_layers",),
    "olmo": ("num_hidden_layers",),
    "olmo2": ("num_hidden_layers",),
    "qwen2": ("num_hidden_layers",),
    "qwen3": ("num_hidden_layers",),
    "falcon": ("num_hidden_layers",),
    "gpt2": ("n_layer",),
    "gpt_bigcode": ("n_layer",),
    "gpt_neox": ("num_hidden_layers",),
    "bloom": ("n_layer",),
    "opt": ("num_hidden_layers",),
    "bert": ("num_hidden_layers",),
    "roberta": ("num_hidden_layers",),
    "distilbert": ("n_layers",),
}


class TransformationContractError(ValueError):
    """Raised when a generated transformation cannot be verified exactly."""


class UnsupportedTransformationError(TransformationContractError):
    """Raised for edit families without a verifier-grade replay contract."""


@dataclass(frozen=True)
class TransformationCheckpointContract:
    """Explicit raw-transformation identity and declared layer topology.

    This deliberately extends, rather than changes, the shared pruning
    contract.  Pruning has a separately-versioned target protocol; raw
    transformations need this extra binding because their scopes can select
    one exact layer.
    """

    model_type: str
    architecture: str
    config_sha256: str
    layer_count: int


@dataclass(frozen=True)
class TransformationScope:
    """A parsed target scope with optional, canonical layer qualifiers."""

    base_scope: str
    layer_limit: int | None = None
    layer: int | None = None

    @property
    def canonical(self) -> str:
        qualifiers: list[str] = []
        if self.layer_limit is not None:
            qualifiers.append(f"layers={self.layer_limit}")
        if self.layer is not None:
            qualifiers.append(f"layer={self.layer}")
        if not qualifiers:
            return self.base_scope
        return f"{self.base_scope}@{','.join(qualifiers)}"


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normal_edit_type(edit_type: object) -> str:
    if not isinstance(edit_type, str) or not edit_type:
        raise TransformationContractError("edit type must be a non-empty string")
    if edit_type in UNSUPPORTED_VERIFIER_GRADE_EDIT_TYPES:
        raise UnsupportedTransformationError(
            f"{edit_type} has no verifier-grade generated-lane contract; "
            "implement a dedicated storage and replay contract first"
        )
    if edit_type not in VERIFIER_GRADE_GENERATED_EDIT_TYPES:
        raise UnsupportedTransformationError(
            f"{edit_type!r} is not a verifier-grade generated transformation"
        )
    return edit_type


def transformation_algorithm(edit_type: object) -> str:
    """Return the immutable algorithm identifier for a supported edit family."""

    return _ALGORITHMS[_normal_edit_type(edit_type)]


def _require_exact_parameter_keys(
    parameters: object,
    *,
    expected: frozenset[str],
    edit_type: str,
) -> Mapping[str, object]:
    if not isinstance(parameters, Mapping):
        raise TransformationContractError(
            f"{edit_type} parameters must be a JSON object"
        )
    keys = set(parameters)
    if keys != expected or not all(isinstance(key, str) for key in keys):
        raise TransformationContractError(
            f"{edit_type} parameters must contain exactly {sorted(expected)}"
        )
    return parameters


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TransformationContractError(f"{field} must be a positive integer")
    return value


def _positive_finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransformationContractError(f"{field} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise TransformationContractError(f"{field} must be a finite positive number")
    return normalized


def canonical_transformation_parameters(
    edit_type: object,
    parameters: object,
) -> dict[str, int | float]:
    """Validate an edit's exact parameters and return their canonical form.

    The returned object is intentionally strict: it contains no aliases,
    defaults, or ignored values that a generator could interpret differently.
    """

    normalized_edit_type = _normal_edit_type(edit_type)
    if normalized_edit_type == QUANT_RTN:
        payload = _require_exact_parameter_keys(
            parameters,
            expected=frozenset({"bits", "group_size"}),
            edit_type=normalized_edit_type,
        )
        bits = _positive_int(payload["bits"], field="quant_rtn.bits")
        if not 2 <= bits <= 8:
            raise TransformationContractError("quant_rtn.bits must be in [2, 8]")
        return {
            "bits": bits,
            "group_size": _positive_int(
                payload["group_size"], field="quant_rtn.group_size"
            ),
        }

    if normalized_edit_type == SYNTHETIC_LOWRANK_DELTA:
        payload = _require_exact_parameter_keys(
            parameters,
            expected=frozenset({"rank", "scale"}),
            edit_type=normalized_edit_type,
        )
        rank = _positive_int(payload["rank"], field="synthetic_lowrank_delta.rank")
        if rank > MAX_SYNTHETIC_LOWRANK_RANK:
            raise TransformationContractError(
                "synthetic_lowrank_delta.rank must not exceed "
                f"{MAX_SYNTHETIC_LOWRANK_RANK}"
            )
        return {
            "rank": rank,
            "scale": _positive_finite_float(
                payload["scale"], field="synthetic_lowrank_delta.scale"
            ),
        }

    if normalized_edit_type == SYNTHETIC_DENSE_UPDATE:
        payload = _require_exact_parameter_keys(
            parameters,
            expected=frozenset({"step_size", "iterations"}),
            edit_type=normalized_edit_type,
        )
        iterations = _positive_int(
            payload["iterations"], field="synthetic_dense_update.iterations"
        )
        if iterations > MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS:
            raise TransformationContractError(
                "synthetic_dense_update.iterations must not exceed "
                f"{MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS}"
            )
        return {
            "step_size": _positive_finite_float(
                payload["step_size"], field="synthetic_dense_update.step_size"
            ),
            "iterations": iterations,
        }

    raise AssertionError(f"unhandled supported edit type: {normalized_edit_type}")


def canonical_transformation_spec(
    edit_type: object,
    parameters: object,
) -> dict[str, object]:
    """Return a versioned, canonical transformation specification."""

    normalized_edit_type = _normal_edit_type(edit_type)
    return {
        "schema": TRANSFORMATION_PARAMETERS_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "edit_type": normalized_edit_type,
        "algorithm": transformation_algorithm(normalized_edit_type),
        "parameters": canonical_transformation_parameters(
            normalized_edit_type, parameters
        ),
    }


def parse_transformation_scope(raw_scope: object) -> TransformationScope:
    """Parse only the supported target grammar.

    ``ffn``, ``attn``, and ``all`` can be qualified by ``layers=N`` (the first
    N layers) and/or ``layer=N`` (one exact layer).  Qualifiers are normalized
    into a stable order so they can be manifest-bound.  A malformed qualifier
    is an error rather than a permissive fallback to the unqualified scope.
    """

    if not isinstance(raw_scope, str):
        raise TransformationContractError("transformation scope must be a string")
    text = raw_scope.strip()
    if not text or text.count("@") > 1:
        raise TransformationContractError("transformation scope syntax is invalid")
    has_qualifiers = "@" in text
    if has_qualifiers:
        raw_base, raw_qualifiers = text.split("@", 1)
    else:
        raw_base, raw_qualifiers = text, ""
    base_scope = raw_base.strip().lower()
    if base_scope not in TRANSFORMATION_SCOPES:
        raise TransformationContractError(
            f"transformation scope must begin with one of {sorted(TRANSFORMATION_SCOPES)}"
        )
    if not has_qualifiers:
        return TransformationScope(base_scope=base_scope)
    if not raw_qualifiers.strip():
        raise TransformationContractError("transformation scope qualifier is invalid")

    values: dict[str, int] = {}
    for raw_item in raw_qualifiers.split(","):
        item = raw_item.strip()
        if not item or item.count("=") != 1:
            raise TransformationContractError(
                "transformation scope qualifier is invalid"
            )
        raw_name, raw_value = (part.strip() for part in item.split("=", 1))
        name = raw_name.lower()
        if name not in {"layers", "layer"} or name in values:
            raise TransformationContractError(
                "transformation scope qualifier is invalid"
            )
        if not _NONNEGATIVE_DECIMAL_RE.fullmatch(raw_value):
            raise TransformationContractError(
                "transformation scope qualifier is invalid"
            )
        value = int(raw_value)
        if name == "layers" and value == 0:
            raise TransformationContractError(
                "layers qualifier must be greater than zero"
            )
        values[name] = value

    layer_limit = values.get("layers")
    layer = values.get("layer")
    if layer_limit is not None and layer is not None and layer >= layer_limit:
        raise TransformationContractError(
            "layer qualifier must be smaller than the layers qualifier"
        )
    return TransformationScope(
        base_scope=base_scope,
        layer_limit=layer_limit,
        layer=layer,
    )


def validate_transformation_scope(raw_scope: object) -> str:
    """Return the canonical scope string or raise a contract error."""

    return parse_transformation_scope(raw_scope).canonical


def _declared_layer_count(config: Mapping[str, object], *, model_type: str) -> int:
    path = _LAYER_COUNT_CONFIG_PATHS.get(model_type)
    if path is None:  # Defensive: checkpoint_pruning_contract already gates this.
        raise TransformationContractError(
            f"raw transformation has no layer-count policy for model_type={model_type!r}"
        )
    value: object = config
    for segment in path:
        if not isinstance(value, Mapping):
            value = None
            break
        value = value.get(segment)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TransformationContractError(
            "checkpoint config.json must declare a positive "
            f"{'.'.join(path)} for raw transformation target selection"
        )
    return value


def _scope_within_declared_layers(
    scope: TransformationScope,
    *,
    layer_count: int,
) -> TransformationScope:
    if scope.layer_limit is not None and scope.layer_limit > layer_count:
        raise TransformationContractError(
            "transformation layers qualifier exceeds checkpoint declared layer count"
        )
    if scope.layer is not None and scope.layer >= layer_count:
        raise TransformationContractError(
            "transformation layer qualifier is outside checkpoint declared layer count"
        )
    return scope


def validate_transformation_scope_for_contract(
    raw_scope: object,
    *,
    contract: TransformationCheckpointContract,
) -> str:
    """Return a canonical scope proven to be inside the checkpoint topology."""

    return _scope_within_declared_layers(
        parse_transformation_scope(raw_scope), layer_count=contract.layer_count
    ).canonical


def _pruning_contract(
    contract: TransformationCheckpointContract,
) -> PruningCheckpointContract:
    """Use the shared path-role grammar without widening its public contract."""

    return PruningCheckpointContract(
        model_type=contract.model_type,
        architecture=contract.architecture,
        config_sha256=contract.config_sha256,
    )


def checkpoint_transformation_contract(
    checkpoint_dir: Path,
) -> TransformationCheckpointContract:
    """Resolve a safe checkpoint plus its explicit configured layer count."""

    try:
        base = checkpoint_pruning_contract(checkpoint_dir)
    except PruningContractError as exc:
        raise TransformationContractError(str(exc)) from exc
    try:
        config = load_json_object(
            checkpoint_dir / "config.json", label="checkpoint config.json"
        )
    except StrictJsonError as exc:
        raise TransformationContractError(
            f"checkpoint config.json is invalid: {exc}"
        ) from exc
    return TransformationCheckpointContract(
        model_type=base.model_type,
        architecture=base.architecture,
        config_sha256=base.config_sha256,
        layer_count=_declared_layer_count(config, model_type=base.model_type),
    )


def transformation_target_role(
    name: str,
    *,
    contract: TransformationCheckpointContract,
    ndim: int,
) -> str | None:
    """Return the explicit role of a candidate language-model matrix."""

    role = pruning_target_role(name, contract=_pruning_contract(contract), ndim=ndim)
    if role is None:
        return None
    if isinstance(role, str) and role in {"ffn", "attn", "router"}:
        return role
    raise TransformationContractError("shared target resolver returned an invalid role")


def transformation_target_layer(
    name: str,
    *,
    contract: TransformationCheckpointContract,
) -> int | None:
    """Return an unambiguous architecture-specific layer number, if present."""

    patterns = _LAYER_PATTERNS.get(contract.architecture)
    if patterns is None:
        return None
    matches = {
        int(match.group(1)) for pattern in patterns for match in pattern.finditer(name)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def is_transformation_target(
    name: str,
    *,
    scope: object,
    contract: TransformationCheckpointContract,
    ndim: int,
) -> bool:
    """Return whether a tensor belongs to a canonical transformation scope."""

    parsed_scope = _scope_within_declared_layers(
        parse_transformation_scope(scope), layer_count=contract.layer_count
    )
    role = transformation_target_role(name, contract=contract, ndim=ndim)
    if role is None:
        return False
    if parsed_scope.base_scope != "all" and role != parsed_scope.base_scope:
        return False
    layer = transformation_target_layer(name, contract=contract)
    if layer is None or layer >= contract.layer_count:
        return False
    if parsed_scope.layer_limit is not None and layer >= parsed_scope.layer_limit:
        return False
    return parsed_scope.layer is None or layer == parsed_scope.layer


def transformation_target_entry(name: str, tensor: Any) -> dict[str, object]:
    """Return the minimal target descriptor without importing a tensor backend."""

    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": [int(dimension) for dimension in tensor.shape],
        "numel": int(tensor.numel()),
    }


def _canonical_target_descriptor(target: object) -> dict[str, object]:
    if not isinstance(target, Mapping) or set(target) != _TARGET_INPUT_KEYS:
        raise TransformationContractError(
            "transformation target descriptor must contain only name, dtype, shape, and numel"
        )
    name = target["name"]
    dtype = target["dtype"]
    shape = target["shape"]
    numel = target["numel"]
    if not isinstance(name, str) or not name:
        raise TransformationContractError("transformation target name is invalid")
    if not isinstance(dtype, str) or not dtype:
        raise TransformationContractError("transformation target dtype is invalid")
    dtype_lower = dtype.lower()
    if "float" not in dtype_lower or "float8" in dtype_lower or "mxfp" in dtype_lower:
        raise TransformationContractError(
            "transformation targets must use regular floating-point storage"
        )
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not all(
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension > 0
            for dimension in shape
        )
    ):
        raise TransformationContractError("transformation target shape is invalid")
    if isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0:
        raise TransformationContractError("transformation target numel is invalid")
    expected_numel = math.prod(shape)
    if numel != expected_numel:
        raise TransformationContractError(
            "transformation target numel does not match its shape"
        )
    return {"name": name, "dtype": dtype, "shape": list(shape), "numel": numel}


def _validate_rank_for_target(
    *,
    parameters: Mapping[str, int | float],
    shape: list[int],
) -> None:
    if "rank" not in parameters:
        return
    rank = parameters["rank"]
    if not isinstance(rank, int):  # defensive: canonical parameters guarantee this
        raise TransformationContractError("synthetic low-rank rank is invalid")
    maximum_rank = min(shape[0], math.prod(shape[1:]))
    if rank > maximum_rank:
        raise TransformationContractError(
            "synthetic low-rank rank exceeds a selected target's matrix rank"
        )


def transformation_target_manifest(
    *,
    edit_type: object,
    parameters: object,
    scope: object,
    contract: TransformationCheckpointContract,
    targets: Iterable[object],
) -> dict[str, object]:
    """Build a canonical manifest for exactly the selected transformation targets."""

    parsed_scope = _scope_within_declared_layers(
        parse_transformation_scope(scope), layer_count=contract.layer_count
    )
    spec = canonical_transformation_spec(edit_type, parameters)
    canonical_parameters = spec["parameters"]
    if not isinstance(canonical_parameters, Mapping):  # defensive type narrowing
        raise AssertionError("canonical transformation parameters must be a mapping")

    normalized_targets: list[dict[str, object]] = []
    names: set[str] = set()
    for target in targets:
        descriptor = _canonical_target_descriptor(target)
        name = str(descriptor["name"])
        shape = descriptor["shape"]
        if not isinstance(shape, list):  # guaranteed by _canonical_target_descriptor
            raise AssertionError("canonical target shape must be a list")
        ndim = len(shape)
        role = transformation_target_role(name, contract=contract, ndim=ndim)
        if role is None or not is_transformation_target(
            name,
            scope=parsed_scope.canonical,
            contract=contract,
            ndim=ndim,
        ):
            raise TransformationContractError(
                f"target {name!r} is outside the canonical transformation scope"
            )
        layer = transformation_target_layer(name, contract=contract)
        if layer is None or layer >= contract.layer_count:
            raise TransformationContractError(
                f"target {name!r} is outside checkpoint declared layer count"
            )
        if name in names:
            raise TransformationContractError("transformation targets must be unique")
        names.add(name)
        _validate_rank_for_target(parameters=canonical_parameters, shape=shape)
        normalized_targets.append({**descriptor, "role": role, "layer": layer})

    normalized_targets.sort(key=lambda target: str(target["name"]))
    if not normalized_targets:
        raise TransformationContractError(
            "transformation target manifest contains no selected tensors"
        )
    return {
        "schema": TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY_VERSION,
        "edit_type": spec["edit_type"],
        "algorithm": spec["algorithm"],
        "parameters": dict(canonical_parameters),
        "scope": parsed_scope.canonical,
        "model_type": contract.model_type,
        "architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "layer_count": contract.layer_count,
        "targets": normalized_targets,
    }


def _manifest_contract(
    manifest: Mapping[str, object],
) -> TransformationCheckpointContract:
    model_type = manifest.get("model_type")
    architecture = manifest.get("architecture")
    config_sha256 = manifest.get("config_sha256")
    layer_count = manifest.get("layer_count")
    if (
        not isinstance(model_type, str)
        or not isinstance(architecture, str)
        or not isinstance(config_sha256, str)
        or not _SHA256_RE.fullmatch(config_sha256)
        or isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
    ):
        raise TransformationContractError(
            "transformation target manifest checkpoint identity is invalid"
        )
    if MODEL_TYPE_ARCHITECTURES.get(model_type) != architecture:
        raise TransformationContractError(
            "transformation target manifest model architecture is invalid"
        )
    return TransformationCheckpointContract(
        model_type=model_type,
        architecture=architecture,
        config_sha256=config_sha256,
        layer_count=layer_count,
    )


def transformation_target_manifest_sha256(manifest: object) -> str:
    """Return the digest only for an exact canonical target manifest."""

    if not isinstance(manifest, Mapping):
        raise TransformationContractError(
            "transformation target manifest must be an object"
        )
    contract = _manifest_contract(manifest)
    targets = manifest.get("targets")
    if not isinstance(targets, list):
        raise TransformationContractError(
            "transformation target manifest targets must be a list"
        )
    input_targets: list[dict[str, object]] = []
    for target in targets:
        if not isinstance(target, Mapping) or set(target) != _TARGET_MANIFEST_KEYS:
            raise TransformationContractError(
                "transformation target manifest target entry is invalid"
            )
        input_targets.append(
            {
                "name": target["name"],
                "dtype": target["dtype"],
                "shape": target["shape"],
                "numel": target["numel"],
            }
        )
    canonical = transformation_target_manifest(
        edit_type=manifest.get("edit_type"),
        parameters=manifest.get("parameters"),
        scope=manifest.get("scope"),
        contract=contract,
        targets=input_targets,
    )
    if dict(manifest) != canonical:
        raise TransformationContractError(
            "transformation target manifest is not canonical or has unbound fields"
        )
    return _canonical_json_sha256(canonical)


__all__ = [
    "FP8_QUANT",
    "GROUPWISE_RTN_DEQUANTIZED_ALGORITHM",
    "LOWRANK_SVD",
    "QUANT_RTN",
    "SYNTHETIC_DENSE_UPDATE",
    "SYNTHETIC_DENSE_UPDATE_ALGORITHM",
    "SYNTHETIC_LOWRANK_DELTA",
    "SYNTHETIC_LOWRANK_DELTA_ALGORITHM",
    "TRANSFORMATION_CONTRACT_VERSION",
    "TRANSFORMATION_PARAMETERS_SCHEMA",
    "TRANSFORMATION_SCOPE_POLICY_VERSION",
    "TRANSFORMATION_SCOPES",
    "TRANSFORMATION_TARGET_MANIFEST_SCHEMA",
    "TransformationContractError",
    "TransformationCheckpointContract",
    "TransformationScope",
    "UnsupportedTransformationError",
    "UNSUPPORTED_VERIFIER_GRADE_EDIT_TYPES",
    "VERIFIER_GRADE_GENERATED_EDIT_TYPES",
    "canonical_transformation_parameters",
    "canonical_transformation_spec",
    "checkpoint_transformation_contract",
    "is_transformation_target",
    "parse_transformation_scope",
    "transformation_algorithm",
    "transformation_target_entry",
    "transformation_target_layer",
    "transformation_target_manifest",
    "transformation_target_manifest_sha256",
    "transformation_target_role",
    "validate_transformation_scope",
    "validate_transformation_scope_for_contract",
]
