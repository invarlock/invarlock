"""Independent numerical and target-selection oracle for raw transformations.

This module is verifier-owned.  It deliberately does not import the
materializer, its transformation contract, or the pruning target resolver.
The public repaired v1 ABI below is parsed and executed independently so a
self-consistent generator defect cannot certify its own artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

TRANSFORMATION_CONTRACT_VERSION = "verifier-grade-transformation-v1"
TRANSFORMATION_SCOPE_POLICY_VERSION = "architecture-aware-transformation-v1"
TRANSFORMATION_PARAMETERS_SCHEMA = "invarlock/transformation-parameters-v1"
TRANSFORMATION_TARGET_MANIFEST_SCHEMA = "invarlock/transformation-target-manifest-v1"
TRANSFORMATION_REPLAY_SCHEMA = "invarlock/generated-transformation-replay-v1"
TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA = (
    "invarlock/transformation-materialization-v1"
)
TRANSFORMATION_MATERIALIZATION_RECEIPT = "transformation_materialization.json"
CANONICAL_EXECUTION_POLICY = "cpu-float32-or-float64-v1"

GROUPWISE_RTN_DEQUANTIZED_ALGORITHM = "groupwise_rtn_dequantized_per_row_v1"
SYNTHETIC_LOWRANK_DELTA_ALGORITHM = "deterministic_synthetic_lowrank_delta_v1"
SYNTHETIC_DENSE_UPDATE_ALGORITHM = "deterministic_synthetic_dense_update_v1"
MAX_SYNTHETIC_LOWRANK_RANK = 32
MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS = 16

QUANT_RTN = "quant_rtn"
SYNTHETIC_LOWRANK_DELTA = "synthetic_lowrank_delta"
SYNTHETIC_DENSE_UPDATE = "synthetic_dense_update"
_SUPPORTED_EDIT_TYPES = frozenset(
    {QUANT_RTN, SYNTHETIC_LOWRANK_DELTA, SYNTHETIC_DENSE_UPDATE}
)
_UNSUPPORTED_EDIT_TYPES = frozenset({"fp8_quant", "lowrank_svd"})
_SCOPES = frozenset({"ffn", "attn", "all"})
_ROW_CHUNK_SIZE = 256


class TransformationOracleError(ValueError):
    """Raised when a raw artifact is outside the independently replayable ABI."""


@dataclass(frozen=True)
class OracleCheckpointContract:
    """Oracle-owned checkpoint identity used only for target resolution."""

    model_type: str
    architecture: str
    config_sha256: str
    layer_count: int


@dataclass(frozen=True)
class _Scope:
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
        return (
            self.base_scope
            if not qualifiers
            else f"{self.base_scope}@{','.join(qualifiers)}"
        )


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MODEL_TYPE_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_NONNEGATIVE_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
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

# This is intentionally duplicated from the materializer-facing contract.
# The raw verifier must derive the configured topology independently rather
# than importing a generator helper or inferring it from checkpoint keys.
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


def _canonical_json_sha256(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TransformationOracleError("canonical JSON value is invalid") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise TransformationOracleError(f"JSON object has duplicate key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise TransformationOracleError(f"JSON constant is not permitted: {value}")


def _read_config(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TransformationOracleError,
    ) as exc:
        raise TransformationOracleError(
            f"checkpoint config.json is invalid: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TransformationOracleError("checkpoint config.json must be an object")
    return payload


def _normal_model_type(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransformationOracleError(
            "checkpoint config.json must declare model_type"
        )
    model_type = value.strip().lower().replace("-", "_")
    if _MODEL_TYPE_RE.fullmatch(model_type) is None:
        raise TransformationOracleError("checkpoint config.json model_type is invalid")
    return model_type


def _declared_layer_count(config: Mapping[str, object], *, model_type: str) -> int:
    path = _LAYER_COUNT_CONFIG_PATHS.get(model_type)
    if path is None:  # Defensive: the architecture table is already explicit.
        raise TransformationOracleError(
            f"raw transformation has no layer-count policy for model_type={model_type!r}"
        )
    value: object = config
    for segment in path:
        if not isinstance(value, Mapping):
            value = None
            break
        value = value.get(segment)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TransformationOracleError(
            "checkpoint config.json must declare a positive "
            f"{'.'.join(path)} for raw transformation target selection"
        )
    return value


def _scope_within_declared_layers(scope: _Scope, *, layer_count: int) -> _Scope:
    if scope.layer_limit is not None and scope.layer_limit > layer_count:
        raise TransformationOracleError(
            "transformation layers qualifier exceeds checkpoint declared layer count"
        )
    if scope.layer is not None and scope.layer >= layer_count:
        raise TransformationOracleError(
            "transformation layer qualifier is outside checkpoint declared layer count"
        )
    return scope


def checkpoint_contract(checkpoint_dir: Path) -> OracleCheckpointContract:
    """Parse a checkpoint configuration with the oracle's own policy."""

    config = _read_config(checkpoint_dir / "config.json")
    model_type = _normal_model_type(config.get("model_type"))
    if model_type == "gpt_oss":
        raise TransformationOracleError(
            "raw transformation does not support GPT-OSS/MXFP4 storage"
        )
    if config.get("quantization_config") not in (None, {}):
        raise TransformationOracleError(
            "raw transformation requires an unquantized floating-point checkpoint"
        )
    try:
        config_text = json.dumps(config, allow_nan=False, sort_keys=True).lower()
    except (TypeError, ValueError) as exc:
        raise TransformationOracleError("checkpoint config.json is invalid") from exc
    if "mxfp4" in config_text:
        raise TransformationOracleError(
            "raw transformation does not support MXFP4 storage"
        )
    architecture = _MODEL_TYPE_ARCHITECTURES.get(model_type)
    if architecture is None:
        raise TransformationOracleError(
            "raw transformation has no independent target resolver for "
            f"model_type={model_type!r}"
        )
    return OracleCheckpointContract(
        model_type=model_type,
        architecture=architecture,
        config_sha256=_canonical_json_sha256(config),
        layer_count=_declared_layer_count(config, model_type=model_type),
    )


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TransformationOracleError(f"{field} must be a positive integer")
    return value


def _positive_finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransformationOracleError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TransformationOracleError(f"{field} must be a finite positive number")
    return result


def _parameters_for(edit_type: str, parameters: object) -> dict[str, int | float]:
    if not isinstance(parameters, Mapping):
        raise TransformationOracleError(f"{edit_type} parameters must be a JSON object")
    if edit_type == QUANT_RTN:
        expected = {"bits", "group_size"}
        if set(parameters) != expected:
            raise TransformationOracleError(
                "quant_rtn parameters must contain exactly ['bits', 'group_size']"
            )
        bits = _positive_int(parameters["bits"], field="quant_rtn.bits")
        if not 2 <= bits <= 8:
            raise TransformationOracleError("quant_rtn.bits must be in [2, 8]")
        return {
            "bits": bits,
            "group_size": _positive_int(
                parameters["group_size"], field="quant_rtn.group_size"
            ),
        }
    if edit_type == SYNTHETIC_LOWRANK_DELTA:
        expected = {"rank", "scale"}
        if set(parameters) != expected:
            raise TransformationOracleError(
                "synthetic_lowrank_delta parameters must contain exactly ['rank', 'scale']"
            )
        rank = _positive_int(parameters["rank"], field="synthetic_lowrank_delta.rank")
        if rank > MAX_SYNTHETIC_LOWRANK_RANK:
            raise TransformationOracleError(
                "synthetic_lowrank_delta.rank must not exceed "
                f"{MAX_SYNTHETIC_LOWRANK_RANK}"
            )
        return {
            "rank": rank,
            "scale": _positive_finite_float(
                parameters["scale"], field="synthetic_lowrank_delta.scale"
            ),
        }
    if edit_type == SYNTHETIC_DENSE_UPDATE:
        expected = {"step_size", "iterations"}
        if set(parameters) != expected:
            raise TransformationOracleError(
                "synthetic_dense_update parameters must contain exactly ['iterations', 'step_size']"
            )
        iterations = _positive_int(
            parameters["iterations"], field="synthetic_dense_update.iterations"
        )
        if iterations > MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS:
            raise TransformationOracleError(
                "synthetic_dense_update.iterations must not exceed "
                f"{MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS}"
            )
        return {
            "step_size": _positive_finite_float(
                parameters["step_size"], field="synthetic_dense_update.step_size"
            ),
            "iterations": iterations,
        }
    raise AssertionError(f"unhandled edit type: {edit_type}")


def canonical_transformation_spec(
    edit_type: object, parameters: object
) -> dict[str, object]:
    """Parse the repaired v1 ABI without consulting the materializer contract."""

    if not isinstance(edit_type, str) or not edit_type:
        raise TransformationOracleError("edit type must be a non-empty string")
    if edit_type in _UNSUPPORTED_EDIT_TYPES:
        raise TransformationOracleError(
            f"{edit_type} has no verifier-grade generated-lane contract; "
            "implement an independent raw-transformation ABI first"
        )
    if edit_type not in _SUPPORTED_EDIT_TYPES:
        raise TransformationOracleError(
            f"{edit_type!r} is not an independently replayable raw transformation"
        )
    algorithms = {
        QUANT_RTN: GROUPWISE_RTN_DEQUANTIZED_ALGORITHM,
        SYNTHETIC_LOWRANK_DELTA: SYNTHETIC_LOWRANK_DELTA_ALGORITHM,
        SYNTHETIC_DENSE_UPDATE: SYNTHETIC_DENSE_UPDATE_ALGORITHM,
    }
    return {
        "schema": TRANSFORMATION_PARAMETERS_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "edit_type": edit_type,
        "algorithm": algorithms[edit_type],
        "parameters": _parameters_for(edit_type, parameters),
    }


def _parse_scope(value: object) -> _Scope:
    if not isinstance(value, str):
        raise TransformationOracleError("transformation scope must be a string")
    text = value.strip()
    if not text or text.count("@") > 1:
        raise TransformationOracleError("transformation scope syntax is invalid")
    raw_base, separator, raw_qualifiers = text.partition("@")
    base_scope = raw_base.strip().lower()
    if base_scope not in _SCOPES:
        raise TransformationOracleError(
            "transformation scope must begin with one of ['all', 'attn', 'ffn']"
        )
    if not separator:
        return _Scope(base_scope=base_scope)
    if not raw_qualifiers.strip():
        raise TransformationOracleError("transformation scope qualifier is invalid")
    qualifiers: dict[str, int] = {}
    for raw_item in raw_qualifiers.split(","):
        item = raw_item.strip()
        if not item or item.count("=") != 1:
            raise TransformationOracleError("transformation scope qualifier is invalid")
        raw_name, raw_number = (part.strip() for part in item.split("=", 1))
        name = raw_name.lower()
        if name not in {"layers", "layer"} or name in qualifiers:
            raise TransformationOracleError("transformation scope qualifier is invalid")
        if _NONNEGATIVE_DECIMAL_RE.fullmatch(raw_number) is None:
            raise TransformationOracleError("transformation scope qualifier is invalid")
        number = int(raw_number)
        if name == "layers" and number == 0:
            raise TransformationOracleError(
                "layers qualifier must be greater than zero"
            )
        qualifiers[name] = number
    layer_limit = qualifiers.get("layers")
    layer = qualifiers.get("layer")
    if layer_limit is not None and layer is not None and layer >= layer_limit:
        raise TransformationOracleError(
            "layer qualifier must be smaller than the layers qualifier"
        )
    return _Scope(base_scope=base_scope, layer_limit=layer_limit, layer=layer)


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


def _target_role(
    name: str, *, contract: OracleCheckpointContract, ndim: int
) -> str | None:
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
    raise AssertionError(f"unknown oracle architecture: {architecture}")


def _target_layer(name: str, *, contract: OracleCheckpointContract) -> int | None:
    patterns = _LAYER_PATTERNS[contract.architecture]
    matches = {
        int(match.group(1)) for pattern in patterns for match in pattern.finditer(name)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _is_regular_float_tensor(tensor: torch.Tensor) -> None:
    if tensor.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise TransformationOracleError(
            "transformation targets must use float16, bfloat16, float32, or float64 storage"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise TransformationOracleError("transformation input tensor is non-finite")
    if tensor.dim() < 2 or tensor.shape[0] <= 0 or math.prod(tensor.shape[1:]) <= 0:
        raise TransformationOracleError("transformation target shape is empty")


def _compute_dtype(tensor: torch.Tensor) -> torch.dtype:
    _is_regular_float_tensor(tensor)
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _reference_abs_mean(source_2d: torch.Tensor, *, rows: int) -> float:
    partial_sums = [
        float(
            source_2d[start : min(start + _ROW_CHUNK_SIZE, rows)]
            .abs()
            .sum(dtype=torch.float64)
            .item()
        )
        for start in range(0, rows, _ROW_CHUNK_SIZE)
    ]
    total = 0.0
    for partial in partial_sums:
        total += partial
    mean = total / source_2d.numel()
    return mean if math.isfinite(mean) and mean > 0.0 else 1.0


def _reference_rtn(tensor: torch.Tensor, *, bits: int, group_size: int) -> torch.Tensor:
    compute_dtype = _compute_dtype(tensor)
    rows, columns = int(tensor.shape[0]), int(math.prod(tensor.shape[1:]))
    source = tensor.detach().to(device="cpu").contiguous().reshape(rows, columns)
    result = torch.empty_like(source)
    lower_bound = -(2 ** (bits - 1))
    upper_bound = max((2 ** (bits - 1)) - 1, 1)
    width = min(group_size, columns)
    for row in range(rows):
        values = source[row].to(dtype=compute_dtype)
        for begin in range(0, columns, width):
            end = min(begin + width, columns)
            group = values[begin:end]
            scale = torch.clamp(group.abs().amax() / upper_bound, min=1e-10)
            result[row, begin:end] = (
                torch.round(group / scale).clamp(lower_bound, upper_bound) * scale
            ).to(dtype=tensor.dtype)
    return result.reshape(tensor.shape).contiguous()


def _reference_lowrank(
    tensor: torch.Tensor, *, rank: int, scale: float
) -> torch.Tensor:
    compute_dtype = _compute_dtype(tensor)
    rows, columns = int(tensor.shape[0]), int(math.prod(tensor.shape[1:]))
    if rank > min(rows, columns):
        raise TransformationOracleError(
            "synthetic low-rank rank exceeds a selected target's matrix rank"
        )
    source = tensor.detach().to(device="cpu").contiguous().reshape(rows, columns)
    base_scale = _reference_abs_mean(source, rows=rows)
    basis = torch.arange(1, rank + 1, dtype=compute_dtype, device="cpu")
    columns_vector = torch.arange(1, columns + 1, dtype=compute_dtype, device="cpu")
    right = torch.cos(basis[:, None] * columns_vector[None, :] * 0.013)
    right = right / math.sqrt(columns)
    magnitude = (float(scale) / rank) * 0.001 * base_scale
    result = torch.empty_like(source)
    for start in range(0, rows, _ROW_CHUNK_SIZE):
        stop = min(start + _ROW_CHUNK_SIZE, rows)
        row_numbers = torch.arange(
            start + 1, stop + 1, dtype=compute_dtype, device="cpu"
        )
        left = torch.sin(row_numbers[:, None] * basis[None, :] * 0.017)
        left = left / math.sqrt(rows)
        update = torch.zeros((stop - start, columns), dtype=compute_dtype)
        for component in range(rank):
            update = (
                update
                + left[:, component : component + 1] * right[component : component + 1]
            )
        result[start:stop] = (
            source[start:stop].to(dtype=compute_dtype) + update * magnitude
        ).to(dtype=tensor.dtype)
    if not bool(torch.isfinite(result).all().item()):
        raise TransformationOracleError("synthetic low-rank output is non-finite")
    return result.reshape(tensor.shape).contiguous()


def _reference_dense(
    tensor: torch.Tensor, *, step_size: float, iterations: int
) -> torch.Tensor:
    compute_dtype = _compute_dtype(tensor)
    rows, columns = int(tensor.shape[0]), int(math.prod(tensor.shape[1:]))
    result = (
        tensor.detach().to(device="cpu").contiguous().clone().reshape(rows, columns)
    )
    base_scale = _reference_abs_mean(result, rows=rows)
    columns_vector = torch.arange(1, columns + 1, dtype=compute_dtype, device="cpu")
    magnitude = base_scale * float(step_size) * 100.0
    for iteration in range(1, iterations + 1):
        for start in range(0, rows, _ROW_CHUNK_SIZE):
            stop = min(start + _ROW_CHUNK_SIZE, rows)
            row_numbers = torch.arange(
                start + 1, stop + 1, dtype=compute_dtype, device="cpu"
            )[:, None]
            direction = torch.sin(
                row_numbers * columns_vector[None, :] * 0.00031
                + float(iteration) * 0.17
            ) * torch.cos(
                (row_numbers + columns_vector[None, :]) * 0.013
                - float(iteration) * 0.11
            )
            result[start:stop] = (
                result[start:stop].to(dtype=compute_dtype) + direction * magnitude
            ).to(dtype=tensor.dtype)
    if not bool(torch.isfinite(result).all().item()):
        raise TransformationOracleError("synthetic dense-update output is non-finite")
    return result.reshape(tensor.shape).contiguous()


@dataclass(frozen=True)
class TransformationOracle:
    """Independent repaired v1 semantics bound to one baseline tree."""

    spec: dict[str, object]
    scope: _Scope
    contract: OracleCheckpointContract

    @property
    def normalized_scope(self) -> str:
        return self.scope.canonical

    def is_target(self, name: str, tensor: torch.Tensor) -> bool:
        role = _target_role(name, contract=self.contract, ndim=tensor.dim())
        if role is None:
            return False
        if self.scope.base_scope != "all" and role != self.scope.base_scope:
            return False
        layer = _target_layer(name, contract=self.contract)
        if layer is None or layer >= self.contract.layer_count:
            return False
        if self.scope.layer_limit is not None and layer >= self.scope.layer_limit:
            return False
        return self.scope.layer is None or layer == self.scope.layer

    def target_entry(self, name: str, tensor: torch.Tensor) -> dict[str, object]:
        if not self.is_target(name, tensor):
            raise TransformationOracleError(
                f"target {name!r} is outside the independent transformation scope"
            )
        _is_regular_float_tensor(tensor)
        role = _target_role(name, contract=self.contract, ndim=tensor.dim())
        layer = _target_layer(name, contract=self.contract)
        if role is None or layer is None or layer >= self.contract.layer_count:
            raise TransformationOracleError(
                f"target {name!r} has no independent architecture role/layer"
            )
        parameters = self.spec["parameters"]
        assert isinstance(parameters, Mapping)
        if self.spec["edit_type"] == SYNTHETIC_LOWRANK_DELTA:
            rank = parameters["rank"]
            assert isinstance(rank, int)
            if rank > min(int(tensor.shape[0]), int(math.prod(tensor.shape[1:]))):
                raise TransformationOracleError(
                    "synthetic low-rank rank exceeds a selected target's matrix rank"
                )
        return {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": [int(dimension) for dimension in tensor.shape],
            "numel": int(tensor.numel()),
            "role": role,
            "layer": layer,
        }

    def target_manifest(self, targets: list[dict[str, object]]) -> dict[str, object]:
        normalized = sorted(targets, key=lambda target: str(target["name"]))
        names = [target["name"] for target in normalized]
        if not normalized or len(set(names)) != len(names):
            raise TransformationOracleError(
                "independent transformation targets must be unique and non-empty"
            )
        parameters = self.spec["parameters"]
        assert isinstance(parameters, Mapping)
        return {
            "schema": TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
            "contract_version": TRANSFORMATION_CONTRACT_VERSION,
            "scope_policy": TRANSFORMATION_SCOPE_POLICY_VERSION,
            "edit_type": self.spec["edit_type"],
            "algorithm": self.spec["algorithm"],
            "parameters": dict(parameters),
            "scope": self.normalized_scope,
            "model_type": self.contract.model_type,
            "architecture": self.contract.architecture,
            "config_sha256": self.contract.config_sha256,
            "layer_count": self.contract.layer_count,
            "targets": normalized,
        }

    def target_manifest_sha256(self, targets: list[dict[str, object]]) -> str:
        return _canonical_json_sha256(self.target_manifest(targets))

    def replay_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        parameters = self.spec["parameters"]
        edit_type = self.spec["edit_type"]
        assert isinstance(parameters, Mapping)
        assert isinstance(edit_type, str)
        if edit_type == QUANT_RTN:
            return _reference_rtn(
                tensor,
                bits=int(parameters["bits"]),
                group_size=int(parameters["group_size"]),
            )
        if edit_type == SYNTHETIC_LOWRANK_DELTA:
            return _reference_lowrank(
                tensor,
                rank=int(parameters["rank"]),
                scale=float(parameters["scale"]),
            )
        if edit_type == SYNTHETIC_DENSE_UPDATE:
            return _reference_dense(
                tensor,
                step_size=float(parameters["step_size"]),
                iterations=int(parameters["iterations"]),
            )
        raise AssertionError(f"unsupported independent transformation: {edit_type}")


def build_transformation_oracle(
    checkpoint_dir: Path,
    *,
    edit_type: object,
    parameters: object,
    scope: object,
) -> TransformationOracle:
    """Bind one independently parsed ABI to a baseline checkpoint."""

    contract = checkpoint_contract(checkpoint_dir)
    return TransformationOracle(
        spec=canonical_transformation_spec(edit_type, parameters),
        scope=_scope_within_declared_layers(
            _parse_scope(scope), layer_count=contract.layer_count
        ),
        contract=contract,
    )


__all__ = [
    "CANONICAL_EXECUTION_POLICY",
    "GROUPWISE_RTN_DEQUANTIZED_ALGORITHM",
    "OracleCheckpointContract",
    "SYNTHETIC_DENSE_UPDATE_ALGORITHM",
    "SYNTHETIC_LOWRANK_DELTA_ALGORITHM",
    "TRANSFORMATION_CONTRACT_VERSION",
    "TRANSFORMATION_MATERIALIZATION_RECEIPT",
    "TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA",
    "TRANSFORMATION_PARAMETERS_SCHEMA",
    "TRANSFORMATION_REPLAY_SCHEMA",
    "TRANSFORMATION_SCOPE_POLICY_VERSION",
    "TRANSFORMATION_TARGET_MANIFEST_SCHEMA",
    "TransformationOracle",
    "TransformationOracleError",
    "build_transformation_oracle",
    "canonical_transformation_spec",
    "checkpoint_contract",
]
