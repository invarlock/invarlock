"""Runtime helper surfaces for the run execution stack."""

from __future__ import annotations

import builtins
import gc
from ctypes import CDLL, c_int, c_size_t
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


def _malloc_trim() -> bool:
    try:
        libc = CDLL(None)
        trim = getattr(libc, "malloc_trim", None)
        if trim is None:
            return False
        trim.argtypes = [c_size_t]
        trim.restype = c_int
        return bool(trim(0))
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def release_process_memory() -> None:
    """Best-effort process-wide memory trim after heavyweight model work."""
    try:
        gc.collect()
    except (RuntimeError, TypeError, ValueError):
        pass
    try:
        torch_mod = get_torch()
        if torch_mod is not None and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
            torch_mod.cuda.synchronize()
    except (RuntimeError, TypeError, ValueError, AttributeError):
        pass
    try:
        _malloc_trim()
    except (RuntimeError, TypeError, ValueError, AttributeError, OSError):
        pass


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
        del model
        release_process_memory()
    except (ImportError, RuntimeError, TypeError, ValueError, AttributeError):
        # Cleanup should never raise; fallback is to proceed without cache purge.
        return
