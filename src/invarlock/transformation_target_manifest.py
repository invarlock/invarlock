"""Independent, package-owned target policy for raw transformation receipts.

Raw transformation materialization lives under :mod:`scripts`, but evidence
pack verification is shipped from :mod:`src`.  A manifest must therefore not
be trusted merely because its fields cross-link: this module re-derives the
supported model-family, target role, layer, scope membership, and regular
floating-point storage policy without importing the materializer, its
transformation contract, its streaming implementation, or the pruning
contract.

The policy deliberately accepts only the repaired v1 ABI.  Unknown model
families, multimodal/auxiliary branches, and GPT-OSS/MXFP4-style storage are
unsupported rather than best-effort targets.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

TRANSFORMATION_CONTRACT_VERSION = "verifier-grade-transformation-v1"
TRANSFORMATION_SCOPE_POLICY = "architecture-aware-transformation-v1"
TRANSFORMATION_PARAMETERS_SCHEMA = "invarlock/transformation-parameters-v1"
TRANSFORMATION_TARGET_MANIFEST_SCHEMA = "invarlock/transformation-target-manifest-v1"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MODEL_TYPE_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_TENSOR_NAME_RE = re.compile(r"[^\x00\r\n]+\Z")
_NONNEGATIVE_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_TARGET_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "contract_version",
        "scope_policy",
        "edit_type",
        "algorithm",
        "parameters",
        "scope",
        "model_type",
        "architecture",
        "config_sha256",
        "layer_count",
        "targets",
    }
)
_TARGET_FIELDS = frozenset({"name", "dtype", "shape", "numel", "role", "layer"})
_REGULAR_FLOAT_DTYPES = frozenset(
    {"torch.float16", "torch.float32", "torch.float64", "torch.bfloat16"}
)
MAX_SYNTHETIC_LOWRANK_RANK = 32
MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS = 16
_SUPPORTED_SCOPES = frozenset({"ffn", "attn", "all"})
_ALGORITHMS = {
    "quant_rtn": "groupwise_rtn_dequantized_per_row_v1",
    "synthetic_lowrank_delta": "deterministic_synthetic_lowrank_delta_v1",
    "synthetic_dense_update": "deterministic_synthetic_dense_update_v1",
}

_EXCLUDED_PATH_SEGMENTS = frozenset(
    {
        "audio",
        "connector",
        "image",
        "images",
        "mm_projector",
        "multi_modal_projector",
        "multimodal",
        "multi_token_prediction",
        "multi_token_predictor",
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
_MODEL_TYPE_ARCHITECTURES = {
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
_LAYER_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "decoder": (re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),),
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


class TransformationTargetManifestError(ValueError):
    """Raised when a transformation target manifest is outside the v1 ABI."""


@dataclass(frozen=True)
class _Scope:
    base: str
    layer_limit: int | None = None
    layer: int | None = None

    @property
    def canonical(self) -> str:
        qualifiers: list[str] = []
        if self.layer_limit is not None:
            qualifiers.append(f"layers={self.layer_limit}")
        if self.layer is not None:
            qualifiers.append(f"layer={self.layer}")
        return self.base if not qualifiers else f"{self.base}@{','.join(qualifiers)}"


def canonical_json_sha256(value: object) -> str:
    """Return the v1 canonical digest without relying on generator helpers."""

    try:
        encoded = json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TransformationTargetManifestError(
            "target manifest is not canonical JSON"
        ) from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _exact_json_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                _exact_json_value(actual[key], item) for key, item in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _exact_json_value(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _mapping(
    value: object, *, label: str, fields: frozenset[str]
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TransformationTargetManifestError(
            f"{label} has missing or unbound fields"
        )
    if not all(isinstance(key, str) for key in value):
        raise TransformationTargetManifestError(f"{label} keys must be strings")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TransformationTargetManifestError(f"{label} must be a positive integer")
    return value


def _positive_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransformationTargetManifestError(
            f"{label} must be a finite positive number"
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise TransformationTargetManifestError(
            f"{label} must be a finite positive number"
        )
    return normalized


def _canonical_parameters(
    edit_type: object, parameters: object
) -> tuple[str, str, dict[str, int | float]]:
    if not isinstance(edit_type, str) or edit_type not in _ALGORITHMS:
        raise TransformationTargetManifestError(
            "target manifest edit_type has no verifier-grade transformation contract"
        )
    if not isinstance(parameters, Mapping) or not all(
        isinstance(key, str) for key in parameters
    ):
        raise TransformationTargetManifestError(
            "target manifest parameters must be an object"
        )
    if edit_type == "quant_rtn":
        if set(parameters) != {"bits", "group_size"}:
            raise TransformationTargetManifestError(
                "quant_rtn parameters must contain exactly ['bits', 'group_size']"
            )
        bits = _positive_int(parameters["bits"], label="quant_rtn.bits")
        if not 2 <= bits <= 8:
            raise TransformationTargetManifestError("quant_rtn.bits must be in [2, 8]")
        normalized: dict[str, int | float] = {
            "bits": bits,
            "group_size": _positive_int(
                parameters["group_size"], label="quant_rtn.group_size"
            ),
        }
    elif edit_type == "synthetic_lowrank_delta":
        if set(parameters) != {"rank", "scale"}:
            raise TransformationTargetManifestError(
                "synthetic_lowrank_delta parameters must contain exactly ['rank', 'scale']"
            )
        rank = _positive_int(parameters["rank"], label="synthetic_lowrank_delta.rank")
        if rank > MAX_SYNTHETIC_LOWRANK_RANK:
            raise TransformationTargetManifestError(
                "synthetic_lowrank_delta.rank must not exceed "
                f"{MAX_SYNTHETIC_LOWRANK_RANK}"
            )
        normalized = {
            "rank": rank,
            "scale": _positive_float(
                parameters["scale"], label="synthetic_lowrank_delta.scale"
            ),
        }
    else:
        if set(parameters) != {"step_size", "iterations"}:
            raise TransformationTargetManifestError(
                "synthetic_dense_update parameters must contain exactly ['iterations', 'step_size']"
            )
        iterations = _positive_int(
            parameters["iterations"], label="synthetic_dense_update.iterations"
        )
        if iterations > MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS:
            raise TransformationTargetManifestError(
                "synthetic_dense_update.iterations must not exceed "
                f"{MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS}"
            )
        normalized = {
            "step_size": _positive_float(
                parameters["step_size"], label="synthetic_dense_update.step_size"
            ),
            "iterations": iterations,
        }
    return edit_type, _ALGORITHMS[edit_type], normalized


def _parse_scope(value: object) -> _Scope:
    if not isinstance(value, str):
        raise TransformationTargetManifestError(
            "target manifest scope must be a string"
        )
    text = value.strip()
    if not text or text.count("@") > 1:
        raise TransformationTargetManifestError(
            "target manifest scope syntax is invalid"
        )
    raw_base, separator, raw_qualifiers = text.partition("@")
    base = raw_base.strip().lower()
    if base not in _SUPPORTED_SCOPES:
        raise TransformationTargetManifestError("target manifest scope base is invalid")
    if not separator:
        scope = _Scope(base=base)
    else:
        if not raw_qualifiers.strip():
            raise TransformationTargetManifestError(
                "target manifest scope qualifier is invalid"
            )
        qualifiers: dict[str, int] = {}
        for raw_item in raw_qualifiers.split(","):
            item = raw_item.strip()
            if not item or item.count("=") != 1:
                raise TransformationTargetManifestError(
                    "target manifest scope qualifier is invalid"
                )
            raw_name, raw_number = (part.strip() for part in item.split("=", 1))
            name = raw_name.lower()
            if name not in {"layers", "layer"} or name in qualifiers:
                raise TransformationTargetManifestError(
                    "target manifest scope qualifier is invalid"
                )
            if _NONNEGATIVE_DECIMAL_RE.fullmatch(raw_number) is None:
                raise TransformationTargetManifestError(
                    "target manifest scope qualifier is invalid"
                )
            number = int(raw_number)
            if name == "layers" and number == 0:
                raise TransformationTargetManifestError(
                    "target manifest layers qualifier must be greater than zero"
                )
            qualifiers[name] = number
        layer_limit = qualifiers.get("layers")
        layer = qualifiers.get("layer")
        if layer_limit is not None and layer is not None and layer >= layer_limit:
            raise TransformationTargetManifestError(
                "target manifest layer qualifier must be smaller than layers"
            )
        scope = _Scope(base=base, layer_limit=layer_limit, layer=layer)
    if value != scope.canonical:
        raise TransformationTargetManifestError(
            "target manifest scope must use canonical syntax"
        )
    return scope


def _scope_within_declared_layers(scope: _Scope, *, layer_count: int) -> _Scope:
    if scope.layer_limit is not None and scope.layer_limit > layer_count:
        raise TransformationTargetManifestError(
            "target manifest layers qualifier exceeds declared layer count"
        )
    if scope.layer is not None and scope.layer >= layer_count:
        raise TransformationTargetManifestError(
            "target manifest layer qualifier is outside declared layer count"
        )
    return scope


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


def _target_role(name: str, *, architecture: str, ndim: int) -> str | None:
    if ndim < 2 or _is_excluded_path(name):
        return None
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
    raise AssertionError(f"unsupported architecture {architecture!r}")


def _target_layer(name: str, *, architecture: str) -> int | None:
    matches = {
        int(match.group(1))
        for pattern in _LAYER_PATTERNS[architecture]
        for match in pattern.finditer(name)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _normal_model_type(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or not value:
        raise TransformationTargetManifestError(
            "target manifest model_type must be a non-empty string"
        )
    normalized = value.strip().lower().replace("-", "_")
    if value != normalized or _MODEL_TYPE_RE.fullmatch(normalized) is None:
        raise TransformationTargetManifestError(
            "target manifest model_type must be canonical"
        )
    if normalized == "gpt_oss":
        raise TransformationTargetManifestError(
            "raw transformations do not support GPT-OSS/MXFP4 storage"
        )
    architecture = _MODEL_TYPE_ARCHITECTURES.get(normalized)
    if architecture is None:
        raise TransformationTargetManifestError(
            "target manifest model_type has no independent target resolver"
        )
    return normalized, architecture


def _canonical_target(
    value: object,
    *,
    scope: _Scope,
    architecture: str,
    layer_count: int,
    edit_type: str,
    parameters: Mapping[str, int | float],
) -> dict[str, object]:
    target = _mapping(value, label="target manifest target", fields=_TARGET_FIELDS)
    name = target["name"]
    if not isinstance(name, str) or _TENSOR_NAME_RE.fullmatch(name) is None:
        raise TransformationTargetManifestError(
            "target manifest target name is invalid"
        )
    lowered_name = name.lower()
    if "mxfp" in lowered_name or "gpt_oss" in lowered_name:
        raise TransformationTargetManifestError(
            "target manifest identifies unsupported GPT-OSS/MXFP4 storage"
        )
    dtype = target["dtype"]
    if not isinstance(dtype, str) or dtype not in _REGULAR_FLOAT_DTYPES:
        raise TransformationTargetManifestError(
            "target manifest target dtype must be regular floating-point storage"
        )
    shape = target["shape"]
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
        raise TransformationTargetManifestError(
            "target manifest target shape must be a positive matrix shape"
        )
    numel = target["numel"]
    if isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0:
        raise TransformationTargetManifestError(
            "target manifest target numel must be a positive integer"
        )
    if numel != math.prod(shape):
        raise TransformationTargetManifestError(
            "target manifest target numel does not match shape"
        )
    expected_role = _target_role(name, architecture=architecture, ndim=len(shape))
    if expected_role is None:
        raise TransformationTargetManifestError(
            f"target manifest target {name!r} is outside the independent transformation scope"
        )
    role = target["role"]
    if role != expected_role:
        raise TransformationTargetManifestError(
            f"target manifest target {name!r} role does not match its architecture path"
        )
    if scope.base != "all" and role != scope.base:
        raise TransformationTargetManifestError(
            f"target manifest target {name!r} is outside the declared scope"
        )
    if scope.base == "all" and role not in {"ffn", "attn", "router"}:
        raise TransformationTargetManifestError(
            f"target manifest target {name!r} is outside the declared scope"
        )
    expected_layer = _target_layer(name, architecture=architecture)
    layer = target["layer"]
    if (
        expected_layer is None
        or isinstance(layer, bool)
        or not isinstance(layer, int)
        or layer < 0
        or layer != expected_layer
        or expected_layer >= layer_count
    ):
        raise TransformationTargetManifestError(
            f"target manifest target {name!r} layer does not match its architecture path"
        )
    if scope.layer_limit is not None and expected_layer >= scope.layer_limit:
        raise TransformationTargetManifestError(
            f"target manifest target {name!r} is outside the layers qualifier"
        )
    if scope.layer is not None and expected_layer != scope.layer:
        raise TransformationTargetManifestError(
            f"target manifest target {name!r} is outside the layer qualifier"
        )
    if edit_type == "synthetic_lowrank_delta":
        rank = parameters["rank"]
        assert isinstance(rank, int)
        if rank > min(shape[0], math.prod(shape[1:])):
            raise TransformationTargetManifestError(
                "synthetic low-rank rank exceeds a selected target's matrix rank"
            )
    return {
        "name": name,
        "dtype": dtype,
        "shape": list(shape),
        "numel": numel,
        "role": role,
        "layer": layer,
    }


def validate_transformation_target_manifest(manifest: object) -> dict[str, object]:
    """Validate and return one exact, independently-derived v1 target manifest.

    This checks semantic target membership using only the serialized manifest,
    so it remains usable when the original checkpoint is not present at
    evidence-pack verification time.  It is intentionally not a numerical
    replay oracle; artifact values still require the independent raw replay
    verifier.
    """

    payload = _mapping(
        manifest,
        label="transformation target manifest",
        fields=_TARGET_MANIFEST_FIELDS,
    )
    if payload["schema"] != TRANSFORMATION_TARGET_MANIFEST_SCHEMA:
        raise TransformationTargetManifestError(
            "target manifest schema is unrecognized"
        )
    if payload["contract_version"] != TRANSFORMATION_CONTRACT_VERSION:
        raise TransformationTargetManifestError(
            "target manifest contract_version is unrecognized"
        )
    if payload["scope_policy"] != TRANSFORMATION_SCOPE_POLICY:
        raise TransformationTargetManifestError(
            "target manifest scope_policy is unrecognized"
        )
    edit_type, algorithm, parameters = _canonical_parameters(
        payload["edit_type"], payload["parameters"]
    )
    if payload["algorithm"] != algorithm:
        raise TransformationTargetManifestError("target manifest algorithm is invalid")
    scope = _parse_scope(payload["scope"])
    model_type, expected_architecture = _normal_model_type(payload["model_type"])
    architecture = payload["architecture"]
    if architecture != expected_architecture:
        raise TransformationTargetManifestError(
            "target manifest model_type and architecture mismatch"
        )
    config_sha256 = payload["config_sha256"]
    if (
        not isinstance(config_sha256, str)
        or _SHA256_RE.fullmatch(config_sha256) is None
    ):
        raise TransformationTargetManifestError(
            "target manifest config_sha256 is invalid"
        )
    layer_count = _positive_int(
        payload["layer_count"], label="target manifest layer_count"
    )
    scope = _scope_within_declared_layers(scope, layer_count=layer_count)
    raw_targets = payload["targets"]
    if not isinstance(raw_targets, list) or not raw_targets:
        raise TransformationTargetManifestError(
            "target manifest must retain at least one selected tensor"
        )
    targets = [
        _canonical_target(
            item,
            scope=scope,
            architecture=expected_architecture,
            layer_count=layer_count,
            edit_type=edit_type,
            parameters=parameters,
        )
        for item in raw_targets
    ]
    names = [str(target["name"]) for target in targets]
    if names != sorted(names) or len(names) != len(set(names)):
        raise TransformationTargetManifestError(
            "target manifest targets must be sorted and unique"
        )
    canonical: dict[str, object] = {
        "schema": TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "edit_type": edit_type,
        "algorithm": algorithm,
        "parameters": parameters,
        "scope": scope.canonical,
        "model_type": model_type,
        "architecture": expected_architecture,
        "config_sha256": config_sha256,
        "layer_count": layer_count,
        "targets": targets,
    }
    if not _exact_json_value(dict(payload), canonical):
        raise TransformationTargetManifestError("target manifest is not canonical")
    return canonical


def transformation_target_manifest_sha256(manifest: object) -> str:
    """Return a digest only after independent semantic validation."""

    return canonical_json_sha256(validate_transformation_target_manifest(manifest))


__all__ = [
    "TRANSFORMATION_CONTRACT_VERSION",
    "TRANSFORMATION_PARAMETERS_SCHEMA",
    "TRANSFORMATION_SCOPE_POLICY",
    "TRANSFORMATION_TARGET_MANIFEST_SCHEMA",
    "TransformationTargetManifestError",
    "canonical_json_sha256",
    "transformation_target_manifest_sha256",
    "validate_transformation_target_manifest",
]
