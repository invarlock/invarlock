"""
HuggingFace Optimum ONNX Runtime Adapter
=======================================

Minimal adapter to load CPU-friendly ONNX Runtime models via Optimum for
causal language modeling. This enables certification of pre-quantized int8
models published for ONNX/Optimum, when used with `edit: noop`.

Notes
- This adapter targets inference (perplexity/log-loss) and does not expose
  parameter/module traversal; guards that require PyTorch module access may
  be inapplicable for ONNX models.
- Optional dependency: `optimum[onnxruntime]` and `onnxruntime`.
"""

from __future__ import annotations

from typing import Any

from invarlock.core.api import ModelAdapter
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import DependencyError, ModelLoadError


class HF_ORT_CausalLM_Adapter(ModelAdapter):
    """Optimum/ONNXRuntime causal LM adapter.

    Provides a lightweight bridge that loads an ORTModelForCausalLM and
    presents enough interface compatibility for evaluation paths that operate
    on logits (perplexity). Snapshot/restore are not supported and will fall
    back to reload in the CLI runner.
    """

    name = "hf_onnx"

    # --- Loading ---------------------------------------------------------
    def load_model(self, model_id: str, device: str = "cpu", **kwargs: Any):  # type: ignore[override]
        with wrap_errors(
            DependencyError,
            "E203",
            "DEPENDENCY-MISSING: optimum/onnxruntime",
            lambda e: {"dependency": "optimum[onnxruntime]"},
        ):
            from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore

        # Prefer CPUExecutionProvider by default; callers may override via kwargs
        providers = kwargs.pop("providers", None)
        if providers is None:
            providers = ["CPUExecutionProvider"]

        # Trust remote code where necessary; users can set to False via kwargs
        trust_remote_code = kwargs.pop("trust_remote_code", True)

        # Some repos use non-default file names; accept overrides but default to standard
        file_name = kwargs.pop("file_name", None)

        with wrap_errors(
            ModelLoadError,
            "E201",
            "MODEL-LOAD-FAILED: optimum ORTModelForCausalLM",
            lambda e: {"model_id": model_id},
        ):
            model = ORTModelForCausalLM.from_pretrained(
                model_id,
                file_name=file_name,
                provider="CPUExecutionProvider"
                if providers == ["CPUExecutionProvider"]
                else None,
                providers=providers,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        # ORT models manage device internally; return as-is
        return model

    # --- Capability probes ----------------------------------------------
    def can_handle(self, model: Any) -> bool:  # type: ignore[override]
        cls_name = model.__class__.__name__
        mod_name = getattr(model.__class__, "__module__", "")
        if "optimum.onnxruntime" in mod_name and "ORTModel" in cls_name:
            return True
        # Heuristic: object exposes generate and is directly callable
        return hasattr(model, "generate") and callable(model)

    # --- Structure description (best-effort) -----------------------------
    def describe(self, model: Any) -> dict[str, Any]:  # type: ignore[override]
        # Attempt to read from HF config when present
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
            "backend": "onnxruntime",
        }

    # --- Snapshot/restore (unsupported; runner falls back to reload) -----
    def snapshot(self, model: Any) -> bytes:  # type: ignore[override]
        raise NotImplementedError("snapshot not supported for ONNXRuntime models")

    def restore(self, model: Any, blob: bytes) -> None:  # type: ignore[override]
        raise NotImplementedError("restore not supported for ONNXRuntime models")


__all__ = ["HF_ORT_CausalLM_Adapter"]
