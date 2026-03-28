"""Runtime helper surfaces for the run execution stack."""

from __future__ import annotations

import builtins
from types import ModuleType
from typing import Any


def _import_optional_module(name: str) -> Any:
    try:
        return builtins.__import__(name)
    except ImportError:
        return None


def get_psutil() -> Any:
    global psutil
    if psutil is None or isinstance(psutil, ModuleType):
        psutil = _import_optional_module("psutil")
    return psutil


def get_torch() -> Any:
    global torch
    if torch is None or isinstance(torch, ModuleType):
        torch = _import_optional_module("torch")
    return torch


psutil: Any = _import_optional_module("psutil")
torch: Any = _import_optional_module("torch")


def reset_optional_runtime_caches() -> None:
    global psutil, torch
    if psutil is None:
        psutil = _import_optional_module("psutil")
    if torch is None:
        torch = _import_optional_module("torch")


def detect_model_profile(model_id: str, adapter: str | None = None) -> Any:
    from invarlock.model_profile import detect_model_profile as _detect_model_profile

    return _detect_model_profile(model_id=model_id, adapter=adapter)


def resolve_tokenizer(profile: Any) -> tuple[Any, str]:
    from invarlock.model_profile import resolve_tokenizer as _resolve_tokenizer

    return _resolve_tokenizer(profile)


def validate_guard_overhead(*args: Any, **kwargs: Any) -> Any:
    from invarlock.reporting.validate import (
        validate_guard_overhead as _validate_guard_overhead,
    )

    return _validate_guard_overhead(*args, **kwargs)


def free_model_memory(model: object | None) -> None:
    """Best-effort cleanup to release GPU memory for a model object."""
    if model is None:
        return
    try:
        import gc

        torch_mod = get_torch()
        del model
        gc.collect()
        if torch_mod is not None and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
            torch_mod.cuda.synchronize()
    except (ImportError, RuntimeError, TypeError, ValueError, AttributeError):
        # Cleanup should never raise; fallback is to proceed without cache purge.
        return
