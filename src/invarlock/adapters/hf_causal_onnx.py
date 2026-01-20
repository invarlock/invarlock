"""
HuggingFace causal LM adapter (ONNX Runtime backend).
====================================================

Role-based adapter for ONNX Runtime causal LM exports. This is inference-only
and does not support snapshot/restore.
"""

from __future__ import annotations

from .hf_onnx import HF_ORT_CausalLM_Adapter


class HF_Causal_ONNX_Adapter(HF_ORT_CausalLM_Adapter):
    name = "hf_causal_onnx"


__all__ = ["HF_Causal_ONNX_Adapter"]

