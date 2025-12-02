"""
Auto adapter resolution utilities.

These helpers map a model identifier (HF directory or Hub ID) to a
concrete built-in adapter name (hf_gpt2, hf_llama, hf_bert) without
adding a hard dependency on Transformers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _read_local_hf_config(model_id: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read config.json from a local HF directory if present."""
    try:
        p = Path(model_id)
    except Exception:
        return None
    cfg_path = p / "config.json"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _detect_quant_family_from_cfg(cfg: dict[str, Any]) -> str | None:
    """Detect quantization family from a HF config dict.

    Returns one of: 'hf_gptq', 'hf_awq', 'hf_bnb' or None if not detected.
    """
    try:
        q = cfg.get("quantization_config") or {}
        if isinstance(q, dict):
            method = str(q.get("quant_method", q.get("quant_method_full", ""))).lower()
            if any(tok in method for tok in ("gptq",)):
                return "hf_gptq"
            if any(tok in method for tok in ("awq",)):
                return "hf_awq"
            # BitsAndBytes style
            if any(
                str(q.get(k, "")).lower() in {"true", "1"}
                for k in ("load_in_4bit", "load_in_8bit")
            ) or any("bitsandbytes" in str(v).lower() for v in q.values()):
                return "hf_bnb"
    except Exception:
        return None
    return None


def resolve_auto_adapter(
    model_id: str | os.PathLike[str], default: str = "hf_gpt2"
) -> str:
    """Resolve an appropriate built-in adapter name for a model.

    Heuristics:
      - Prefer local config.json (no network). Inspect `model_type` and
        `architectures` to classify LLaMA/Mistral vs BERT vs GPT-like.
      - Fallback to simple name heuristics on the model_id string.
      - Default to `hf_gpt2` when unsure.
    """
    cfg = _read_local_hf_config(model_id)
    model_id_str = str(model_id)

    def _from_cfg(c: dict[str, Any]) -> str | None:
        # Prefer explicit quantization families first
        fam = _detect_quant_family_from_cfg(c)
        if fam:
            return fam
        mt = str(c.get("model_type", "")).lower()
        archs = [str(a) for a in c.get("architectures", []) if isinstance(a, str)]
        arch_blob = " ".join(archs)
        if (
            mt in {"llama", "mistral", "qwen", "yi"}
            or "Llama" in arch_blob
            or "Mistral" in arch_blob
        ):
            return "hf_llama"
        # Treat masked-LM families as BERT-like
        if (
            mt in {"bert", "roberta", "distilbert", "albert", "deberta", "deberta-v2"}
            or "MaskedLM" in arch_blob
        ):
            return "hf_bert"
        # Generic causal LM
        if "CausalLM" in arch_blob or mt in {
            "gpt2",
            "gpt_neox",
            "opt",
            "gptj",
            "gptj8bit",
        }:
            return "hf_gpt2"
        return None

    # If local directory contains ONNX model files, prefer hf_onnx
    try:
        p = Path(model_id)
        if p.exists() and p.is_dir():
            # Common Optimum export names
            onnx_files = [
                "model.onnx",
                "decoder_model.onnx",
                "decoder_with_past_model.onnx",
                "encoder_model.onnx",
            ]
            if any((p / fname).exists() for fname in onnx_files):
                return "hf_onnx"
    except Exception:
        pass

    if isinstance(cfg, dict):
        resolved = _from_cfg(cfg)
        if resolved:
            return resolved

    # String heuristics as last resort
    lower_id = model_id_str.lower()
    # Quantized repo heuristics
    if any(k in lower_id for k in ["gptq", "-gptq", "_gptq"]):
        return "hf_gptq"
    if any(k in lower_id for k in ["awq", "-awq", "_awq"]):
        return "hf_awq"
    if any(
        k in lower_id for k in ["bnb", "bitsandbytes", "-4bit", "-8bit", "4bit", "8bit"]
    ):
        return "hf_bnb"
    if any(k in lower_id for k in ["llama", "mistral", "qwen", "yi"]):
        return "hf_llama"
    if any(k in lower_id for k in ["bert", "roberta", "albert", "deberta"]):
        return "hf_bert"
    return default


def apply_auto_adapter_if_needed(cfg: Any) -> Any:
    """Mutate/clone a InvarLockConfig to resolve adapter:auto → concrete adapter.

    Returns the same config object if no change is needed.
    """
    try:
        adapter = str(getattr(cfg.model, "adapter", ""))
        if adapter.strip().lower() not in {"auto", "hf_auto", "auto_hf"}:
            return cfg
        model_id = str(getattr(cfg.model, "id", ""))
        resolved = resolve_auto_adapter(model_id)
        data = cfg.model_dump()
        data.setdefault("model", {})["adapter"] = resolved
        return cfg.__class__(data)  # re-wrap as InvarLockConfig
    except Exception:
        return cfg


__all__ = ["resolve_auto_adapter", "apply_auto_adapter_if_needed"]
