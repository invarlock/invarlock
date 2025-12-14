"""Helpers for Hugging Face model loading.

Centralizes security- and performance-sensitive defaults used by HF adapters.
"""

from __future__ import annotations

import os
from typing import Any

import torch

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    return None


def resolve_trust_remote_code(
    kwargs: dict[str, Any] | None = None, *, default: bool = False
) -> bool:
    """Resolve trust_remote_code with config override and env opt-in."""
    if kwargs and "trust_remote_code" in kwargs:
        coerced = _coerce_bool(kwargs.get("trust_remote_code"))
        if coerced is not None:
            return coerced

    for env_name in (
        "INVARLOCK_TRUST_REMOTE_CODE",
        "TRUST_REMOTE_CODE_BOOL",
        "ALLOW_REMOTE_CODE",
    ):
        env_val = os.environ.get(env_name)
        coerced = _coerce_bool(env_val)
        if coerced is not None:
            return coerced

    return default


def default_torch_dtype() -> torch.dtype:
    """Pick a safe default dtype for HF loads based on hardware."""
    if torch.cuda.is_available():
        try:
            if (
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ):
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.float16

    return torch.float32


def resolve_torch_dtype(kwargs: dict[str, Any] | None = None) -> torch.dtype | str:
    """Resolve torch_dtype from kwargs or choose a hardware-aware default."""
    if kwargs and "torch_dtype" in kwargs:
        val = kwargs.get("torch_dtype")
        if isinstance(val, torch.dtype):
            return val
        if isinstance(val, str):
            s = val.strip().lower()
            if s == "auto":
                return "auto"
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if s in mapping:
                return mapping[s]

    return default_torch_dtype()


__all__ = ["resolve_trust_remote_code", "default_torch_dtype", "resolve_torch_dtype"]
