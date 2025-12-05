"""
HuggingFace BitsAndBytes Adapter (plugin)
=========================================

Optional adapter for loading 4/8-bit quantized causal LMs via bitsandbytes
through Transformers. Requires GPU for practical use.
Install with the `gpu` extra on supported platforms.

This adapter handles both:
1. Fresh quantization of FP16 models (load_in_8bit/load_in_4bit)
2. Loading pre-quantized BNB checkpoints (auto-detected via quantization_config)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock.adapters.capabilities import (
    ModelCapabilities,
    QuantizationMethod,
    detect_quantization_from_config,
)
from invarlock.adapters.hf_mixin import HFAdapterMixin
from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import DependencyError, ModelLoadError


def _is_local_path(model_id: str) -> bool:
    """Check if model_id is a local filesystem path."""
    return Path(model_id).exists()


def _detect_pre_quantized_bnb(model_id: str) -> tuple[bool, int]:
    """
    Detect if a local checkpoint is pre-quantized with BNB.

    Returns:
        Tuple of (is_pre_quantized, bits) where bits is 8 or 4.
    """
    if not _is_local_path(model_id):
        return False, 0

    config_path = Path(model_id) / "config.json"
    if not config_path.exists():
        return False, 0

    try:
        import json

        config_data = json.loads(config_path.read_text())
        quant_cfg = config_data.get("quantization_config", {})

        if not quant_cfg:
            return False, 0

        # Check for BNB quantization
        quant_method = quant_cfg.get("quant_method", "").lower()
        if quant_method == "bitsandbytes" or "load_in_8bit" in quant_cfg:
            if quant_cfg.get("load_in_8bit"):
                return True, 8
            if quant_cfg.get("load_in_4bit"):
                return True, 4
            # Fallback to bits field
            bits = quant_cfg.get("bits", 8)
            return True, bits

    except Exception:
        pass

    return False, 0


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

        # Check if this is a pre-quantized checkpoint
        is_pre_quantized, pre_quant_bits = _detect_pre_quantized_bnb(model_id)

        if is_pre_quantized:
            # Load pre-quantized checkpoint WITHOUT re-applying quantization
            with wrap_errors(
                ModelLoadError,
                "E201",
                "MODEL-LOAD-FAILED: bitsandbytes/transformers (pre-quantized)",
                lambda e: {"model_id": model_id, "pre_quantized_bits": pre_quant_bits},
            ):
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    # Do NOT pass load_in_8bit/load_in_4bit for pre-quantized
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ("load_in_8bit", "load_in_4bit")
                    },
                )
        else:
            # Fresh quantization of FP16 model
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

        # BNB models handle their own device placement via device_map="auto"
        # Do NOT call .to() on BNB models - it will raise an error
        _ = self._resolve_device(device)  # Validate device string only
        return model

    def get_capabilities(self, model: Any) -> ModelCapabilities:
        """Return capabilities for a BNB-quantized model."""
        config = getattr(model, "config", None)
        if config is not None:
            quant_cfg = detect_quantization_from_config(config)
            if quant_cfg.method == QuantizationMethod.BNB_8BIT:
                return ModelCapabilities.for_bnb_8bit(from_checkpoint=True)
            elif quant_cfg.method == QuantizationMethod.BNB_4BIT:
                return ModelCapabilities.for_bnb_4bit(
                    from_checkpoint=True,
                    double_quant=quant_cfg.double_quant,
                )

        # Default to 8-bit if we can't determine
        return ModelCapabilities.for_bnb_8bit()

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
