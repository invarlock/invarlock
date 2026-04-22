"""
HuggingFace causal LM adapter (decoder-only).
=============================================

Role-based adapter for HuggingFace decoder-only causal language models.

This adapter intentionally avoids model-family naming. It selects a structural
spec at runtime (dense FFN vs MoE vs GPT-2-like blocks) and exposes a stable
`describe()` contract for InvarLock gates and reporting.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from invarlock.core.abi import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import AdapterError, DependencyError, ModelLoadError

from .hf_loading import resolve_core_loader_strategy
from .hf_mixin import HFAdapterMixin

INVARLOCK_CORE_ABI = CORE_ABI

TensorType = torch.Tensor
ModuleType = nn.Module

_ALLOW_DIRECT_SUBMODULE = False


def _first_item(seq: Any) -> Any | None:
    try:
        if hasattr(seq, "__len__") and hasattr(seq, "__getitem__") and len(seq) > 0:
            return seq[0]
    except (TypeError, IndexError, KeyError):
        pass
    try:
        return next(iter(seq))
    except (TypeError, StopIteration):
        return None


def _has_set_attr(obj: Any, name: str) -> bool:
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict) and name in d:
        return True
    if isinstance(obj, nn.Module):
        if hasattr(obj, "_modules") and name in obj._modules:
            return True
        if hasattr(obj, "_parameters") and name in obj._parameters:
            return True
        if hasattr(obj, "_buffers") and name in obj._buffers:
            return True
    return False


def _resolve_norm(layer: Any, *candidates: str) -> tuple[str | None, Any | None]:
    for name in candidates:
        if _has_set_attr(layer, name):
            return name, getattr(layer, name)
    return None, None


def _weight_shape_dim(module: Any, axis: int) -> int | None:
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None:
        return None
    try:
        return int(shape[axis])
    except (IndexError, TypeError, ValueError):
        return None


def _shape_ints(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    dims: list[int] = []
    try:
        for dim in shape:
            dims.append(int(dim))
    except (TypeError, ValueError):
        return None
    return tuple(dims)


def _mixtral_tensorized_moe_parts(layer: Any) -> tuple[Any | None, Any | None]:
    mlp = getattr(layer, "mlp", None)
    gate = getattr(mlp, "gate", None) if mlp is not None else None
    experts = getattr(mlp, "experts", None) if mlp is not None else None
    if gate is None or experts is None:
        return None, None
    if not _has_set_attr(gate, "weight"):
        return None, None
    if not (
        _has_set_attr(experts, "gate_up_proj") and _has_set_attr(experts, "down_proj")
    ):
        return None, None
    return gate, experts


def _coerce_config_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.isdigit():
            return int(stripped)
    return None


def _layer_list(layers: Any) -> list[Any]:
    try:
        return list(layers)
    except TypeError:
        count = len(layers)
        return [layers[idx] for idx in range(int(count))]


def _safe_total_params(model: Any) -> int:
    try:
        return sum(int(param.numel()) for param in model.parameters())
    except (AttributeError, RuntimeError, StopIteration, TypeError, ValueError):
        return 0


def _safe_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, RuntimeError, StopIteration, TypeError):
        return torch.device("cpu")


class _CausalSpec:
    spec_name = "base"

    def matches(self, model: Any, base: Any, layers: Any) -> bool:
        raise NotImplementedError

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        raise NotImplementedError

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
        raise NotImplementedError

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        return {}


class _DenseDecoderSpec(_CausalSpec):
    spec_name = "dense_decoder"

    def matches(self, model: Any, base: Any, layers: Any) -> bool:
        layer = _first_item(layers)
        if layer is None:
            return False
        has_attn = (
            hasattr(layer, "self_attn")
            and _has_set_attr(layer.self_attn, "q_proj")
            and _has_set_attr(layer.self_attn, "k_proj")
            and _has_set_attr(layer.self_attn, "v_proj")
            and _has_set_attr(layer.self_attn, "o_proj")
        )
        has_mlp = (
            hasattr(layer, "mlp")
            and _has_set_attr(layer.mlp, "gate_proj")
            and _has_set_attr(layer.mlp, "up_proj")
            and _has_set_attr(layer.mlp, "down_proj")
        )
        _pre_norm_name, pre_norm = _resolve_norm(
            layer,
            "input_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm",
        )
        _post_norm_name, post_norm = _resolve_norm(
            layer,
            "post_attention_layernorm",
            "pre_attention_layernorm",
        )
        has_norms = pre_norm is not None and post_norm is not None
        return bool(has_attn and has_mlp and has_norms)

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        mlp_dim = int(getattr(config, "intermediate_size", hidden_size * 4) or 0)
        gate_proj = getattr(getattr(layer, "mlp", None), "gate_proj", None)
        gate_proj_dim = _weight_shape_dim(gate_proj, 0)
        if gate_proj_dim is not None:
            mlp_dim = gate_proj_dim
        return int(mlp_dim)

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
        mlp = layer.mlp
        pre_norm_name, pre_norm = _resolve_norm(
            layer,
            "input_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm",
        )
        post_norm_name, post_norm = _resolve_norm(
            layer,
            "post_attention_layernorm",
            "pre_attention_layernorm",
        )
        modules = {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "mlp.gate_proj": mlp.gate_proj,
            "mlp.up_proj": mlp.up_proj,
            "mlp.down_proj": mlp.down_proj,
        }
        if pre_norm is not None:
            modules["input_layernorm"] = pre_norm
            if pre_norm_name and pre_norm_name != "input_layernorm":
                modules[pre_norm_name] = pre_norm
        if post_norm is not None:
            modules["post_attention_layernorm"] = post_norm
            if post_norm_name and post_norm_name != "post_attention_layernorm":
                modules[post_norm_name] = post_norm
        return modules

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        tying: dict[str, str] = {}
        lm_head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
        embed_weight = getattr(getattr(base, "embed_tokens", None), "weight", None)
        if lm_head_weight is not None and lm_head_weight is embed_weight:
            embed_path = "model.embed_tokens.weight"
            outer_model = getattr(model, "model", None)
            if getattr(outer_model, "language_model", None) is base:
                embed_path = "model.language_model.embed_tokens.weight"
            tying["lm_head.weight"] = embed_path
        return tying


class _PhiDecoderSpec(_CausalSpec):
    spec_name = "phi_decoder"

    def matches(self, model: Any, base: Any, layers: Any) -> bool:
        layer = _first_item(layers)
        if layer is None:
            return False
        has_attn = (
            hasattr(layer, "self_attn")
            and _has_set_attr(layer.self_attn, "qkv_proj")
            and _has_set_attr(layer.self_attn, "o_proj")
        )
        has_mlp = (
            hasattr(layer, "mlp")
            and _has_set_attr(layer.mlp, "gate_up_proj")
            and _has_set_attr(layer.mlp, "down_proj")
        )
        has_norms = _has_set_attr(layer, "input_layernorm") and _has_set_attr(
            layer, "post_attention_layernorm"
        )
        return bool(has_attn and has_mlp and has_norms)

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        mlp_dim = int(getattr(config, "intermediate_size", hidden_size * 4) or 0)
        down_proj = getattr(getattr(layer, "mlp", None), "down_proj", None)
        down_proj_dim = _weight_shape_dim(down_proj, 1)
        if down_proj_dim is not None:
            mlp_dim = down_proj_dim
        else:
            gate_up_proj = getattr(getattr(layer, "mlp", None), "gate_up_proj", None)
            gate_up_dim = _weight_shape_dim(gate_up_proj, 0)
            if gate_up_dim is not None:
                mlp_dim = int(gate_up_dim // 2)
        return int(mlp_dim)

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
        mlp = layer.mlp
        return {
            "self_attn.qkv_proj": layer.self_attn.qkv_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "input_layernorm": layer.input_layernorm,
            "post_attention_layernorm": layer.post_attention_layernorm,
            "mlp.gate_up_proj": mlp.gate_up_proj,
            "mlp.down_proj": mlp.down_proj,
        }

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        return _DenseDecoderSpec().tying_map(model, base)


class _GlmDecoderSpec(_CausalSpec):
    spec_name = "glm_decoder"

    def matches(self, model: Any, base: Any, layers: Any) -> bool:
        layer = _first_item(layers)
        if layer is None:
            return False
        has_attn = (
            hasattr(layer, "self_attn")
            and _has_set_attr(layer.self_attn, "q_proj")
            and _has_set_attr(layer.self_attn, "k_proj")
            and _has_set_attr(layer.self_attn, "v_proj")
            and _has_set_attr(layer.self_attn, "o_proj")
        )
        has_mlp = (
            hasattr(layer, "mlp")
            and _has_set_attr(layer.mlp, "gate_up_proj")
            and _has_set_attr(layer.mlp, "down_proj")
        )
        has_norms = _has_set_attr(layer, "input_layernorm") and _has_set_attr(
            layer, "post_attention_layernorm"
        )
        return bool(has_attn and has_mlp and has_norms)

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        mlp_dim = int(getattr(config, "intermediate_size", hidden_size * 4) or 0)
        down_proj = getattr(getattr(layer, "mlp", None), "down_proj", None)
        down_proj_dim = _weight_shape_dim(down_proj, 1)
        if down_proj_dim is not None:
            mlp_dim = down_proj_dim
        return int(mlp_dim)

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
        return {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "mlp.gate_up_proj": layer.mlp.gate_up_proj,
            "mlp.down_proj": layer.mlp.down_proj,
            "input_layernorm": layer.input_layernorm,
            "post_attention_layernorm": layer.post_attention_layernorm,
        }

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        tying: dict[str, str] = {}
        lm_head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
        embed_weight = getattr(getattr(base, "embed_tokens", None), "weight", None)
        if lm_head_weight is not None and lm_head_weight is embed_weight:
            tying["lm_head.weight"] = "model.embed_tokens.weight"
        return tying


class _Qwen35LinearDecoderSpec(_CausalSpec):
    spec_name = "qwen35_linear_decoder"

    def matches(self, model: Any, base: Any, layers: Any) -> bool:
        layer = _first_item(layers)
        if layer is None:
            return False
        has_attn = (
            hasattr(layer, "linear_attn")
            and _has_set_attr(layer.linear_attn, "in_proj_qkv")
            and _has_set_attr(layer.linear_attn, "out_proj")
        )
        has_mlp = (
            hasattr(layer, "mlp")
            and _has_set_attr(layer.mlp, "gate_proj")
            and _has_set_attr(layer.mlp, "up_proj")
            and _has_set_attr(layer.mlp, "down_proj")
        )
        has_norms = _has_set_attr(layer, "input_layernorm") and _has_set_attr(
            layer, "post_attention_layernorm"
        )
        return bool(has_attn and has_mlp and has_norms)

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        return _DenseDecoderSpec().infer_mlp_dim(layer, config, hidden_size)

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
        mlp = layer.mlp
        return {
            "linear_attn.in_proj_qkv": layer.linear_attn.in_proj_qkv,
            "linear_attn.out_proj": layer.linear_attn.out_proj,
            "input_layernorm": layer.input_layernorm,
            "post_attention_layernorm": layer.post_attention_layernorm,
            "mlp.gate_proj": mlp.gate_proj,
            "mlp.up_proj": mlp.up_proj,
            "mlp.down_proj": mlp.down_proj,
        }

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        return _DenseDecoderSpec().tying_map(model, base)


class _MoEDecoderSpec(_CausalSpec):
    spec_name = "moe_decoder"

    def matches(self, model: Any, base: Any, layers: Any) -> bool:
        layer = _first_item(layers)
        if layer is None:
            return False
        has_attn = (
            hasattr(layer, "self_attn")
            and _has_set_attr(layer.self_attn, "q_proj")
            and _has_set_attr(layer.self_attn, "k_proj")
            and _has_set_attr(layer.self_attn, "v_proj")
            and _has_set_attr(layer.self_attn, "o_proj")
        )
        moe = getattr(layer, "block_sparse_moe", None)
        experts = getattr(moe, "experts", None) if moe is not None else None
        expert0 = _first_item(experts) if experts is not None else None
        has_legacy_moe = bool(
            expert0 is not None
            and _has_set_attr(expert0, "w1")
            and _has_set_attr(expert0, "w2")
        )
        mixtral_gate, mixtral_experts = _mixtral_tensorized_moe_parts(layer)
        has_tensorized_moe = bool(
            mixtral_gate is not None and mixtral_experts is not None
        )
        has_norms = _has_set_attr(layer, "input_layernorm") and _has_set_attr(
            layer, "post_attention_layernorm"
        )
        return bool(has_attn and (has_legacy_moe or has_tensorized_moe) and has_norms)

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        mlp_dim = int(getattr(config, "intermediate_size", hidden_size * 4) or 0)
        moe = getattr(layer, "block_sparse_moe", None)
        experts = getattr(moe, "experts", None) if moe is not None else None
        expert0 = _first_item(experts) if experts is not None else None
        if expert0 is not None:
            w1_dim = _weight_shape_dim(getattr(expert0, "w1", None), 0)
            if w1_dim is not None:
                mlp_dim = w1_dim
                return int(mlp_dim)
        _mixtral_gate, mixtral_experts = _mixtral_tensorized_moe_parts(layer)
        if mixtral_experts is not None:
            intermediate_dim = getattr(mixtral_experts, "intermediate_dim", None)
            if isinstance(intermediate_dim, int) and intermediate_dim > 0:
                return int(intermediate_dim)
            intermediate_size = getattr(mixtral_experts, "intermediate_size", None)
            if isinstance(intermediate_size, int) and intermediate_size > 0:
                return int(intermediate_size)
            gate_up_shape = _shape_ints(getattr(mixtral_experts, "gate_up_proj", None))
            if (
                gate_up_shape is not None
                and len(gate_up_shape) >= 2
                and gate_up_shape[-2] > 0
            ):
                return int(gate_up_shape[-2] // 2)
        return int(mlp_dim)

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
        mixtral_gate, mixtral_experts = _mixtral_tensorized_moe_parts(layer)
        if mixtral_gate is not None and mixtral_experts is not None:
            return {
                "self_attn.q_proj": layer.self_attn.q_proj,
                "self_attn.k_proj": layer.self_attn.k_proj,
                "self_attn.v_proj": layer.self_attn.v_proj,
                "self_attn.o_proj": layer.self_attn.o_proj,
                "input_layernorm": layer.input_layernorm,
                "post_attention_layernorm": layer.post_attention_layernorm,
                "mlp.router": mixtral_gate,
                "mlp.gate": mixtral_gate,
                "mlp.experts": mixtral_experts,
            }
        moe = layer.block_sparse_moe
        expert0 = _first_item(moe.experts)
        if expert0 is None:
            raise AdapterError(
                code="E202",
                message="ADAPTER-STRUCTURE-INVALID: MoE layer missing experts",
                details={"layer_class": layer.__class__.__name__},
            )
        return {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "input_layernorm": layer.input_layernorm,
            "post_attention_layernorm": layer.post_attention_layernorm,
            # Best-effort mapping to dense naming used elsewhere in the stack.
            "mlp.gate_proj": expert0.w1,
            "mlp.up_proj": getattr(expert0, "w3", expert0.w1),
            "mlp.down_proj": expert0.w2,
        }

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        return _DenseDecoderSpec().tying_map(model, base)


class _GptOssMoEDecoderSpec(_CausalSpec):
    spec_name = "gpt_oss_moe_decoder"

    def matches(self, model: Any, base: Any, layers: Any) -> bool:
        layer = _first_item(layers)
        if layer is None:
            return False
        has_attn = (
            hasattr(layer, "self_attn")
            and _has_set_attr(layer.self_attn, "q_proj")
            and _has_set_attr(layer.self_attn, "k_proj")
            and _has_set_attr(layer.self_attn, "v_proj")
            and _has_set_attr(layer.self_attn, "o_proj")
        )
        mlp = getattr(layer, "mlp", None)
        router = getattr(mlp, "router", None) if mlp is not None else None
        experts = getattr(mlp, "experts", None) if mlp is not None else None
        has_moe = bool(
            router is not None
            and _has_set_attr(router, "weight")
            and experts is not None
            and _has_set_attr(experts, "gate_up_proj")
            and _has_set_attr(experts, "down_proj")
        )
        has_norms = _has_set_attr(layer, "input_layernorm") and _has_set_attr(
            layer, "post_attention_layernorm"
        )
        return bool(has_attn and has_moe and has_norms)

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        mlp_dim = int(getattr(config, "intermediate_size", hidden_size * 4) or 0)
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is not None:
            intermediate_size = getattr(experts, "intermediate_size", None)
            if isinstance(intermediate_size, int) and intermediate_size > 0:
                return int(intermediate_size)
            shape = _shape_ints(getattr(experts, "gate_up_proj", None))
            if shape is not None and len(shape) >= 3 and shape[-1] > 0:
                return int(shape[-1] // 2)
        return int(mlp_dim)

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
        mlp = layer.mlp
        return {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "input_layernorm": layer.input_layernorm,
            "post_attention_layernorm": layer.post_attention_layernorm,
            "mlp.router": mlp.router,
            "mlp.experts": mlp.experts,
        }

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        return _DenseDecoderSpec().tying_map(model, base)


class _NeoXDecoderSpec(_CausalSpec):
    spec_name = "neox_decoder"

    def matches(self, model: Any, base: Any, layers: Any) -> bool:
        layer = _first_item(layers)
        if layer is None:
            return False
        has_attn = (
            hasattr(layer, "attention")
            and _has_set_attr(layer.attention, "query_key_value")
            and _has_set_attr(layer.attention, "dense")
        )
        has_mlp = (
            hasattr(layer, "mlp")
            and _has_set_attr(layer.mlp, "dense_h_to_4h")
            and _has_set_attr(layer.mlp, "dense_4h_to_h")
        )
        has_norms = _has_set_attr(layer, "input_layernorm") and _has_set_attr(
            layer, "post_attention_layernorm"
        )
        return bool(has_attn and has_mlp and has_norms)

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        mlp_dim = int(getattr(config, "intermediate_size", hidden_size * 4) or 0)
        dense_h_to_4h = getattr(getattr(layer, "mlp", None), "dense_h_to_4h", None)
        dense_dim = _weight_shape_dim(dense_h_to_4h, 0)
        if dense_dim is not None:
            mlp_dim = dense_dim
        return int(mlp_dim)

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
        return {
            "attention.query_key_value": layer.attention.query_key_value,
            "attention.dense": layer.attention.dense,
            "attn.c_attn": layer.attention.query_key_value,
            "attn.c_proj": layer.attention.dense,
            "mlp.dense_h_to_4h": layer.mlp.dense_h_to_4h,
            "mlp.dense_4h_to_h": layer.mlp.dense_4h_to_h,
            "mlp.c_fc": layer.mlp.dense_h_to_4h,
            "mlp.c_proj": layer.mlp.dense_4h_to_h,
            "input_layernorm": layer.input_layernorm,
            "post_attention_layernorm": layer.post_attention_layernorm,
        }

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        tying: dict[str, str] = {}
        lm_head_weight = getattr(getattr(model, "embed_out", None), "weight", None)
        embed_weight = getattr(getattr(base, "embed_in", None), "weight", None)
        if lm_head_weight is not None and lm_head_weight is embed_weight:
            tying["embed_out.weight"] = "gpt_neox.embed_in.weight"
        return tying


class _FalconDecoderSpec(_CausalSpec):
    spec_name = "falcon_decoder"

    def matches(self, model: Any, base: Any, layers: Any) -> bool:
        layer = _first_item(layers)
        if layer is None:
            return False
        has_attn = (
            hasattr(layer, "self_attention")
            and _has_set_attr(layer.self_attention, "query_key_value")
            and _has_set_attr(layer.self_attention, "dense")
        )
        has_mlp = (
            hasattr(layer, "mlp")
            and _has_set_attr(layer.mlp, "dense_h_to_4h")
            and _has_set_attr(layer.mlp, "dense_4h_to_h")
        )
        return bool(has_attn and has_mlp and _has_set_attr(layer, "input_layernorm"))

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        mlp_dim = int(getattr(config, "hidden_size", hidden_size) * 4)
        dense_h_to_4h = getattr(getattr(layer, "mlp", None), "dense_h_to_4h", None)
        dense_dim = _weight_shape_dim(dense_h_to_4h, 0)
        if dense_dim is not None:
            mlp_dim = dense_dim
        return int(mlp_dim)

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
        return {
            "self_attention.query_key_value": layer.self_attention.query_key_value,
            "self_attention.dense": layer.self_attention.dense,
            "attn.c_attn": layer.self_attention.query_key_value,
            "attn.c_proj": layer.self_attention.dense,
            "mlp.dense_h_to_4h": layer.mlp.dense_h_to_4h,
            "mlp.dense_4h_to_h": layer.mlp.dense_4h_to_h,
            "mlp.c_fc": layer.mlp.dense_h_to_4h,
            "mlp.c_proj": layer.mlp.dense_4h_to_h,
            "input_layernorm": layer.input_layernorm,
        }

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        tying: dict[str, str] = {}
        lm_head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
        embed_weight = getattr(getattr(base, "word_embeddings", None), "weight", None)
        if lm_head_weight is not None and lm_head_weight is embed_weight:
            tying["lm_head.weight"] = "transformer.word_embeddings.weight"
        return tying


class _OptDecoderSpec(_CausalSpec):
    spec_name = "opt_decoder"

    def matches(self, model: Any, base: Any, layers: Any) -> bool:
        layer = _first_item(layers)
        if layer is None:
            return False
        has_attn = (
            hasattr(layer, "self_attn")
            and _has_set_attr(layer.self_attn, "q_proj")
            and _has_set_attr(layer.self_attn, "k_proj")
            and _has_set_attr(layer.self_attn, "v_proj")
            and _has_set_attr(layer.self_attn, "out_proj")
        )
        has_mlp = _has_set_attr(layer, "fc1") and _has_set_attr(layer, "fc2")
        has_norms = _has_set_attr(layer, "self_attn_layer_norm") and _has_set_attr(
            layer, "final_layer_norm"
        )
        return bool(has_attn and has_mlp and has_norms)

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        mlp_dim = int(
            getattr(
                config,
                "ffn_dim",
                getattr(config, "intermediate_size", hidden_size * 4),
            )
            or 0
        )
        fc1_dim = _weight_shape_dim(getattr(layer, "fc1", None), 0)
        if fc1_dim is not None:
            mlp_dim = fc1_dim
        return int(mlp_dim)

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
        return {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.out_proj": layer.self_attn.out_proj,
            "self_attn.o_proj": layer.self_attn.out_proj,
            "attn.c_proj": layer.self_attn.out_proj,
            "mlp.fc1": layer.fc1,
            "mlp.fc2": layer.fc2,
            "mlp.c_fc": layer.fc1,
            "mlp.c_proj": layer.fc2,
            "input_layernorm": layer.self_attn_layer_norm,
            "self_attn_layer_norm": layer.self_attn_layer_norm,
            "post_attention_layernorm": layer.final_layer_norm,
            "final_layer_norm": layer.final_layer_norm,
        }

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        tying: dict[str, str] = {}
        lm_head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
        decoder = getattr(getattr(model, "model", None), "decoder", None)
        embed_weight = getattr(getattr(decoder, "embed_tokens", None), "weight", None)
        if lm_head_weight is not None and lm_head_weight is embed_weight:
            tying["lm_head.weight"] = "model.decoder.embed_tokens.weight"
        return tying


class _GPT2LikeDecoderSpec(_CausalSpec):
    spec_name = "gpt2_like"

    def matches(self, model: Any, base: Any, layers: Any) -> bool:
        layer = _first_item(layers)
        if layer is None:
            return False
        return bool(
            hasattr(layer, "attn")
            and hasattr(layer.attn, "c_proj")
            and hasattr(layer, "mlp")
            and hasattr(layer.mlp, "c_proj")
        )

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        c_fc = getattr(getattr(layer, "mlp", None), "c_fc", None)
        if c_fc is not None:
            # HF GPT-style uses Conv1D where nf is out_features.
            nf_value = _coerce_config_int(getattr(c_fc, "nf", None))
            if nf_value is not None:
                return nf_value
            c_fc_dim = _weight_shape_dim(c_fc, 0)
            if c_fc_dim is not None:
                return c_fc_dim
        return int(getattr(config, "n_inner", hidden_size * 4) or 0)

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
        return {
            "attn.c_attn": layer.attn.c_attn,
            "attn.c_proj": layer.attn.c_proj,
            "mlp.c_fc": layer.mlp.c_fc,
            "mlp.c_proj": layer.mlp.c_proj,
            "ln_1": layer.ln_1,
            "ln_2": layer.ln_2,
        }

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        tying: dict[str, str] = {}
        lm_head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
        wte_weight = getattr(getattr(base, "wte", None), "weight", None)
        if lm_head_weight is not None and lm_head_weight is wte_weight:
            tying["lm_head.weight"] = "transformer.wte.weight"
        return tying


_SPECS: list[_CausalSpec] = [
    _MoEDecoderSpec(),
    _GptOssMoEDecoderSpec(),
    _PhiDecoderSpec(),
    _GlmDecoderSpec(),
    _Qwen35LinearDecoderSpec(),
    _NeoXDecoderSpec(),
    _FalconDecoderSpec(),
    _OptDecoderSpec(),
    _DenseDecoderSpec(),
    _GPT2LikeDecoderSpec(),
]


class HF_Causal_Adapter(HFAdapterMixin, ModelAdapter):
    """Spec-driven adapter for decoder-only causal LMs."""

    name = "hf_causal"

    def load_model(
        self, model_id: str, device: str = "auto", **kwargs: Any
    ) -> ModuleType | Any:
        try:
            with wrap_errors(
                DependencyError,
                "E203",
                "DEPENDENCY-MISSING: transformers",
                lambda e: {"dependency": "transformers"},
            ):
                strategy = resolve_core_loader_strategy(
                    task="causal",
                    model_id=model_id,
                    kwargs=kwargs,
                    allow_direct_submodule=_ALLOW_DIRECT_SUBMODULE,
                )
                direct_strategy = (
                    resolve_core_loader_strategy(
                        task="causal",
                        model_id=model_id,
                        # Preserve the direct-submodule fallback even when the
                        # initial auto path was attempted with trust_remote_code.
                        kwargs={},
                        allow_direct_submodule=True,
                    )
                    if strategy.strategy == "auto"
                    else strategy
                )
                auto_strategy = (
                    strategy
                    if strategy.strategy == "auto"
                    else resolve_core_loader_strategy(
                        task="causal",
                        model_id=model_id,
                        kwargs=kwargs,
                        allow_direct_submodule=False,
                    )
                )
            self._last_loader_strategy = strategy.strategy
            self._last_loader_label = strategy.loader_label

            try:
                with wrap_errors(
                    ModelLoadError,
                    "E201",
                    f"MODEL-LOAD-FAILED: {strategy.loader_label}",
                    lambda e: {"model_id": model_id},
                ):
                    model = self._load_pretrained_model(
                        strategy.loader,
                        model_id,
                        **kwargs,
                    )
            except ModelLoadError:
                if (
                    strategy.strategy == "auto"
                    and direct_strategy.strategy == "direct_submodule"
                ):
                    self._last_loader_strategy = direct_strategy.strategy
                    self._last_loader_label = direct_strategy.loader_label
                    with wrap_errors(
                        ModelLoadError,
                        "E201",
                        f"MODEL-LOAD-FAILED: {direct_strategy.loader_label}",
                        lambda e: {"model_id": model_id},
                    ):
                        model = self._load_pretrained_model(
                            direct_strategy.loader,
                            model_id,
                            **kwargs,
                        )
                    return self._safe_to_device(model, device)
                if strategy.strategy == "auto":
                    raise
                self._last_loader_strategy = auto_strategy.strategy
                self._last_loader_label = auto_strategy.loader_label
                with wrap_errors(
                    ModelLoadError,
                    "E201",
                    f"MODEL-LOAD-FAILED: {auto_strategy.loader_label}",
                    lambda e: {"model_id": model_id},
                ):
                    model = self._load_pretrained_model(
                        auto_strategy.loader,
                        model_id,
                        **kwargs,
                    )

            return self._safe_to_device(model, device)
        except DependencyError:
            raise

    def _unwrap(self, model: Any) -> tuple[Any, Any, Any]:
        config = getattr(model, "config", None)
        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            decoder = getattr(model.model, "decoder", None)
            if decoder is not None and hasattr(decoder, "layers"):
                return decoder, decoder.layers, config
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            language_model = getattr(model.model, "language_model", None)
            if language_model is not None and hasattr(language_model, "layers"):
                return language_model, language_model.layers, config
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model, model.model.layers, config
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return model.gpt_neox, model.gpt_neox.layers, config
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer, model.transformer.h, config
        if hasattr(model, "layers"):
            return model, model.layers, config
        if hasattr(model, "h"):
            return model, model.h, config
        raise AdapterError(
            code="E202",
            message="ADAPTER-STRUCTURE-INVALID: unrecognized HF causal LM structure",
            details={"model_class": model.__class__.__name__},
        )

    def _select_spec(self, model: Any, base: Any, layers: Any) -> _CausalSpec:
        for spec in _SPECS:
            try:
                if spec.matches(model, base, layers):
                    return spec
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
        raise AdapterError(
            code="E202",
            message="ADAPTER-STRUCTURE-INVALID: no matching HF causal adapter spec",
            details={"model_class": model.__class__.__name__},
        )

    def can_handle(self, model: ModuleType | Any) -> bool:
        try:
            base, layers, _cfg = self._unwrap(model)
        except AdapterError:
            return False
        for spec in _SPECS:
            try:
                if spec.matches(model, base, layers):
                    return True
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
        return False

    def describe(self, model: ModuleType | Any) -> dict[str, Any]:
        base, layers, config = self._unwrap(model)
        if config is None:
            raise AdapterError(
                code="E202",
                message="ADAPTER-STRUCTURE-INVALID: missing HuggingFace config on model",
                details={"model_class": model.__class__.__name__},
            )

        layer_list = _layer_list(layers)
        n_layers = len(layer_list)

        text_config = getattr(config, "text_config", None)

        def _cfg_int(*names: str) -> int | None:
            for container in (config, text_config):
                if container is None:
                    continue
                for name in names:
                    value = _coerce_config_int(getattr(container, name, None))
                    if value is not None:
                        return value
            return None

        n_heads = _cfg_int("num_attention_heads", "n_head")
        hidden_size = _cfg_int("hidden_size", "n_embd")
        vocab_size = _cfg_int("vocab_size")

        if n_heads is None or hidden_size is None:
            raise AdapterError(
                code="E202",
                message="ADAPTER-STRUCTURE-INVALID: missing head/hidden size metadata",
                details={"model_class": model.__class__.__name__},
            )

        spec = self._select_spec(model, base, layers)

        heads_per_layer = [int(n_heads)] * int(n_layers)
        mlp_dims: list[int] = []
        for layer in layer_list:
            mlp_dims.append(spec.infer_mlp_dim(layer, config, int(hidden_size)))

        tying = spec.tying_map(model, base)

        total_params = _safe_total_params(model)
        device = _safe_model_device(model)

        return {
            "n_layer": int(n_layers),
            "heads_per_layer": heads_per_layer,
            "mlp_dims": mlp_dims,
            "tying": tying,
            "model_type": str(getattr(config, "model_type", "") or "causal"),
            "model_class": model.__class__.__name__,
            "hf_model_type": str(getattr(config, "model_type", "") or ""),
            "hf_config_class": config.__class__.__name__
            if hasattr(config, "__class__")
            else "unknown",
            "n_heads": int(n_heads),
            "hidden_size": int(hidden_size),
            "vocab_size": int(vocab_size) if vocab_size is not None else None,
            "total_params": int(total_params),
            "device": str(device),
            "spec": spec.spec_name,
        }

    def get_layer_modules(
        self, model: ModuleType | Any, layer_idx: int
    ) -> dict[str, Any]:
        base, layers, _cfg = self._unwrap(model)
        spec = self._select_spec(model, base, layers)
        layer = layers[layer_idx]
        return spec.layer_modules(model, layer)
