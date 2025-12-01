"""
HuggingFace GPTQ Adapter (plugin)
=================================

Optional adapter for loading AutoGPTQ-quantized causal LMs from the Hub.
Requires the `auto-gptq` extra on supported platforms (typically Linux/CUDA).
"""

from __future__ import annotations

from typing import Any

from invarlock.adapters.hf_mixin import HFAdapterMixin
from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import DependencyError, ModelLoadError


class HF_GPTQ_Adapter(HFAdapterMixin, ModelAdapter):
    name = "hf_gptq"

    # ---- Lifecycle ----
    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any):
        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: auto_gptq/transformers",
            lambda e: {"dependency": "auto_gptq"},
        ):
            try:
                from auto_gptq import AutoGPTQForCausalLM
            except Exception:  # pragma: no cover - import path variations
                from transformers import (
                    AutoModelForCausalLM as AutoGPTQForCausalLM,
                )

        with wrap_errors(
            ModelLoadError,
            "E201",
            "MODEL-LOAD-FAILED: gptq",
            lambda e: {"model_id": model_id},
        ):
            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                trust_remote_code=True,
                inject_fused_attention=False,
                **{k: v for k, v in kwargs.items() if k not in {"device"}},
            )

        return model.to(self._resolve_device(device))

    # ---- Introspection ----
    def can_handle(self, model: Any) -> bool:
        # Heuristic: quantized causal LM typically exposes .config with hidden layers
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

    # ---- Snapshots ----
    # Inherit snapshot()/restore() from HFAdapterMixin


__all__ = ["HF_GPTQ_Adapter"]
