"""
HuggingFace AWQ Adapter (plugin)
================================

Optional adapter for loading AWQ-quantized causal LMs from the Hub.
Requires the `autoawq` extra on supported platforms (typically Linux/CUDA).
"""

from __future__ import annotations

from typing import Any

from invarlock.adapters.hf_mixin import HFAdapterMixin
from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import DependencyError, ModelLoadError


class HF_AWQ_Adapter(HFAdapterMixin, ModelAdapter):
    name = "hf_awq"

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any):
        # Try common import paths used by AWQ projects
        AutoAWQForCausalLM = None
        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: autoawq/awq",
            lambda e: {"dependency": "autoawq/awq"},
        ):
            for mod_path, attr in (
                ("autoawq", "AutoAWQForCausalLM"),
                ("awq", "AutoAWQForCausalLM"),
            ):
                try:  # pragma: no cover - exercised in integration
                    mod = __import__(mod_path, fromlist=[attr])
                    AutoAWQForCausalLM = getattr(mod, attr)
                    break
                except Exception:
                    continue

        if AutoAWQForCausalLM is None:  # pragma: no cover
            # wrap_errors above will have raised; this is a safety
            raise DependencyError(
                code="E203", message="DEPENDENCY-MISSING: autoawq/awq"
            )

        with wrap_errors(
            ModelLoadError,
            "E201",
            "MODEL-LOAD-FAILED: awq",
            lambda e: {"model_id": model_id},
        ):
            model = AutoAWQForCausalLM.from_quantized(
                model_id,
                trust_remote_code=True,
                **{k: v for k, v in kwargs.items() if k != "device"},
            )
        return model.to(self._resolve_device(device))

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


__all__ = ["HF_AWQ_Adapter"]
