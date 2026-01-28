"""
HuggingFace causal LM adapter (decoder-only).
=============================================

Role-based adapter for HuggingFace decoder-only causal language models.

This adapter intentionally avoids model-family naming. It selects a structural
spec at runtime (dense FFN vs MoE vs GPT-2-like blocks) and exposes a stable
`describe()` contract for InvarLock gates and reporting.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import AdapterError, DependencyError, ModelLoadError

from .hf_mixin import HFAdapterMixin

TensorType = torch.Tensor
ModuleType = nn.Module

LIGHT_IMPORT = os.getenv("INVARLOCK_LIGHT_IMPORT", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _first_item(seq: Any) -> Any | None:
    try:
        if hasattr(seq, "__len__") and len(seq) > 0:  # type: ignore[arg-type]
            return seq[0]  # type: ignore[index]
    except Exception:
        pass
    try:
        return next(iter(seq))
    except Exception:
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
        has_norms = _has_set_attr(layer, "input_layernorm") and _has_set_attr(
            layer, "post_attention_layernorm"
        )
        return bool(has_attn and has_mlp and has_norms)

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        mlp_dim = int(getattr(config, "intermediate_size", hidden_size * 4) or 0)
        try:
            gate_proj = getattr(getattr(layer, "mlp", None), "gate_proj", None)
            if gate_proj is not None and hasattr(gate_proj, "weight"):
                mlp_dim = int(gate_proj.weight.shape[0])
        except Exception:
            pass
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
            "mlp.gate_proj": mlp.gate_proj,
            "mlp.up_proj": mlp.up_proj,
            "mlp.down_proj": mlp.down_proj,
        }

    def tying_map(self, model: Any, base: Any) -> dict[str, str]:
        tying: dict[str, str] = {}
        try:
            if hasattr(model, "lm_head") and hasattr(base, "embed_tokens"):
                if model.lm_head.weight is base.embed_tokens.weight:
                    tying["lm_head.weight"] = "model.embed_tokens.weight"
        except Exception:
            pass
        return tying


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
        has_moe = bool(
            expert0 is not None
            and _has_set_attr(expert0, "w1")
            and _has_set_attr(expert0, "w2")
        )
        has_norms = _has_set_attr(layer, "input_layernorm") and _has_set_attr(
            layer, "post_attention_layernorm"
        )
        return bool(has_attn and has_moe and has_norms)

    def infer_mlp_dim(self, layer: Any, config: Any, hidden_size: int) -> int:
        mlp_dim = int(getattr(config, "intermediate_size", hidden_size * 4) or 0)
        try:
            moe = getattr(layer, "block_sparse_moe", None)
            experts = getattr(moe, "experts", None) if moe is not None else None
            expert0 = _first_item(experts) if experts is not None else None
            if expert0 is not None:
                w1 = getattr(expert0, "w1", None)
                if w1 is not None and hasattr(w1, "weight"):
                    mlp_dim = int(w1.weight.shape[0])
        except Exception:
            pass
        return int(mlp_dim)

    def layer_modules(self, model: Any, layer: Any) -> dict[str, Any]:
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
        try:
            c_fc = getattr(getattr(layer, "mlp", None), "c_fc", None)
            if c_fc is not None and hasattr(c_fc, "weight"):
                # HF GPT-style uses Conv1D where nf is out_features.
                if hasattr(c_fc, "nf"):
                    return int(c_fc.nf)
                return int(c_fc.weight.shape[0])
        except Exception:
            pass
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
        try:
            if hasattr(model, "lm_head") and hasattr(base, "wte"):
                if model.lm_head.weight is base.wte.weight:
                    tying["lm_head.weight"] = "transformer.wte.weight"
        except Exception:
            pass
        return tying


_SPECS: list[_CausalSpec] = [
    _MoEDecoderSpec(),
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
                from transformers import AutoModelForCausalLM  # type: ignore

            with wrap_errors(
                ModelLoadError,
                "E201",
                "MODEL-LOAD-FAILED: transformers AutoModelForCausalLM",
                lambda e: {"model_id": model_id},
            ):
                model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

            return self._safe_to_device(model, device)
        except DependencyError:
            if LIGHT_IMPORT:
                stub = SimpleNamespace(name="hf_causal_stub")
                stub.to = lambda *_a, **_k: stub  # type: ignore[attr-defined]
                return stub
            raise

    def _unwrap(self, model: Any) -> tuple[Any, Any, Any]:
        config = getattr(model, "config", None)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model, model.model.layers, config
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
            except Exception:
                continue
        return _DenseDecoderSpec()

    def can_handle(self, model: ModuleType | Any) -> bool:
        try:
            base, layers, _cfg = self._unwrap(model)
        except Exception:
            return False
        return any(spec.matches(model, base, layers) for spec in _SPECS)

    def describe(self, model: ModuleType | Any) -> dict[str, Any]:
        base, layers, config = self._unwrap(model)
        if config is None:
            raise AdapterError(
                code="E202",
                message="ADAPTER-STRUCTURE-INVALID: missing HuggingFace config on model",
                details={"model_class": model.__class__.__name__},
            )

        try:
            n_layers = len(layers)
        except Exception:
            n_layers = sum(1 for _ in iter(layers))

        def _coerce_int(value: Any) -> int | None:
            try:
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
            except Exception:
                return None
            return None

        n_heads = _coerce_int(getattr(config, "num_attention_heads", None))
        if n_heads is None:
            n_heads = _coerce_int(getattr(config, "n_head", None))

        hidden_size = _coerce_int(getattr(config, "hidden_size", None))
        if hidden_size is None:
            hidden_size = _coerce_int(getattr(config, "n_embd", None))

        vocab_size = _coerce_int(getattr(config, "vocab_size", None))

        if n_heads is None or hidden_size is None:
            raise AdapterError(
                code="E202",
                message="ADAPTER-STRUCTURE-INVALID: missing head/hidden size metadata",
                details={"model_class": model.__class__.__name__},
            )

        spec = self._select_spec(model, base, layers)

        heads_per_layer = [int(n_heads)] * int(n_layers)
        mlp_dims: list[int] = []
        for idx in range(int(n_layers)):
            layer = layers[idx]
            mlp_dims.append(spec.infer_mlp_dim(layer, config, int(hidden_size)))

        tying = spec.tying_map(model, base)

        total_params = 0
        try:
            total_params = sum(p.numel() for p in model.parameters())
        except Exception:
            total_params = 0

        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

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
