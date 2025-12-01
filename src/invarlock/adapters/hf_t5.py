"""
HuggingFace T5 Model Adapter
============================

ModelAdapter implementation for HuggingFace T5 encoder-decoder models.

Loads AutoModelForSeq2SeqLM (e.g., t5-small/base/large) and exposes a minimal
describe() sufficient for guard policies and reporting.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import AdapterError, DependencyError, ModelLoadError

from .hf_mixin import HFAdapterMixin

TensorType = torch.Tensor
ModuleType = nn.Module


class HF_T5_Adapter(HFAdapterMixin, ModelAdapter):
    """HuggingFace T5 adapter using AutoModelForSeq2SeqLM."""

    name = "hf_t5"

    def load_model(self, model_id: str, device: str = "auto") -> ModuleType | Any:  # type: ignore[override]
        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: transformers",
            lambda e: {"dependency": "transformers"},
        ):
            from transformers import AutoModelForSeq2SeqLM  # type: ignore

        with wrap_errors(
            ModelLoadError,
            "E201",
            "MODEL-LOAD-FAILED: transformers AutoModelForSeq2SeqLM",
            lambda e: {"model_id": model_id},
        ):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        return model.to(self._resolve_device(device))

    def can_handle(self, model: ModuleType | Any) -> bool:  # type: ignore[override]
        cfg = getattr(model, "config", None)
        if cfg is None:
            return False
        try:
            mt = str(getattr(cfg, "model_type", "")).lower()
        except Exception:
            mt = ""
        if mt == "t5":
            return True
        # Heuristic: encoder-decoder with lm_head and shared embedding
        return bool(
            getattr(cfg, "is_encoder_decoder", False)
            and hasattr(model, "lm_head")
            and hasattr(model, "shared")
        )

    def describe(self, model: ModuleType | Any) -> dict[str, Any]:  # type: ignore[override]
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise AdapterError(
                code="E202",
                message="ADAPTER-STRUCTURE-INVALID: missing HuggingFace config on model",
                details={"model_class": model.__class__.__name__},
            )

        # Extract key dimensions with safe fallbacks
        enc_layers = int(
            getattr(cfg, "num_layers", getattr(cfg, "num_encoder_layers", 0)) or 0
        )
        dec_layers = int(getattr(cfg, "num_decoder_layers", enc_layers) or 0)
        n_layer = int(enc_layers + dec_layers or max(enc_layers, dec_layers))
        n_heads = int(
            getattr(cfg, "num_heads", getattr(cfg, "num_attention_heads", 0)) or 0
        )
        d_model = int(getattr(cfg, "d_model", getattr(cfg, "hidden_size", 0)) or 0)
        d_ff = int(getattr(cfg, "d_ff", (d_model * 4) if d_model else 0) or 0)
        vocab_size = int(getattr(cfg, "vocab_size", 0) or 0)

        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        heads_per_layer = [n_heads] * max(1, n_layer)
        mlp_dims = [d_ff] * max(1, n_layer)

        # Weight tying: T5 ties lm_head ↔ shared embedding
        tying_map: dict[str, str] = {}
        try:
            if hasattr(model, "lm_head") and hasattr(model, "shared"):
                if getattr(model.lm_head, "weight", None) is getattr(
                    model.shared, "weight", object()
                ):
                    tying_map["lm_head.weight"] = "shared.weight"
        except Exception:
            pass

        total_params = 0
        try:
            total_params = sum(p.numel() for p in model.parameters())
        except Exception:
            total_params = 0

        return {
            "n_layer": int(max(1, n_layer)),
            "heads_per_layer": heads_per_layer,
            "mlp_dims": mlp_dims,
            "tying": tying_map,
            "model_type": "t5",
            "model_class": model.__class__.__name__,
            "n_heads": n_heads,
            "hidden_size": d_model,
            "vocab_size": vocab_size,
            "total_params": total_params,
            "device": str(device),
        }

    # snapshot/restore provided by HFAdapterMixin
    def snapshot(self, model: ModuleType) -> bytes:  # type: ignore[override]
        return super().snapshot(model)

    def restore(self, model: ModuleType, blob: bytes) -> None:  # type: ignore[override]
        return super().restore(model, blob)


__all__ = ["HF_T5_Adapter"]
