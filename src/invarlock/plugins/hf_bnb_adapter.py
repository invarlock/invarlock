"""
HuggingFace BitsAndBytes Adapter (plugin)
=========================================

Optional adapter for loading 4/8-bit quantized causal LMs via bitsandbytes
through Transformers. Requires GPU for practical use.
Install with the `gpu` extra on supported platforms.
"""

from __future__ import annotations

from typing import Any

from invarlock.adapters.hf_mixin import HFAdapterMixin
from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import DependencyError, ModelLoadError


class HF_BNB_Adapter(HFAdapterMixin, ModelAdapter):
    name = "hf_bnb"

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any):
        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: transformers",
            lambda e: {"dependency": "transformers"},
        ):
            from transformers import AutoModelForCausalLM

        # Default to 8-bit if not specified
        load_in_8bit = bool(kwargs.pop("load_in_8bit", True))
        load_in_4bit = bool(kwargs.pop("load_in_4bit", False))

        if load_in_4bit:
            load_in_8bit = False

        with wrap_errors(
            ModelLoadError,
            "E201",
            "MODEL-LOAD-FAILED: bitsandbytes/transformers",
            lambda e: {"model_id": model_id},
        ):
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                trust_remote_code=True,
                **kwargs,
            )

        # Let HF device map place on CUDA if available; ensure device object resolves
        _ = self._resolve_device(device)
        return model

    def can_handle(self, model: Any) -> bool:
        cfg = getattr(model, "config", None)
        return hasattr(cfg, "n_layer") or hasattr(cfg, "num_hidden_layers")

    def describe(self, model: Any) -> dict[str, Any]:
        cfg = getattr(model, "config", None)
        n_layer = int(
            getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)) or 0
        )
        n_head = int(
            getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0)) or 0
        )
        heads = [n_head] * n_layer if n_layer and n_head else []
        return {
            "n_layer": n_layer,
            "heads_per_layer": heads,
            "mlp_dims": [],
            "tying": {},
        }


__all__ = ["HF_BNB_Adapter"]
