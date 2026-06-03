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

from invarlock.core import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import AdapterError, DependencyError, ModelLoadError

from .hf_causal_specs import (
    _SPECS,
    _CausalSpec,
    _coerce_config_int,
    _layer_list,
    _safe_model_device,
    _safe_total_params,
)
from .hf_loading import resolve_core_loader_strategy
from .hf_mixin import HFAdapterMixin

INVARLOCK_CORE_ABI = CORE_ABI

TensorType = torch.Tensor
ModuleType = nn.Module

_ALLOW_DIRECT_SUBMODULE = False


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
        if hasattr(model, "model") and hasattr(model.model, "model"):
            nested_model = getattr(model.model, "model", None)
            if nested_model is not None and hasattr(nested_model, "layers"):
                return nested_model, nested_model.layers, config
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
