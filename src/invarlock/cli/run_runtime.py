"""Runtime helper surfaces for the run execution stack."""

from __future__ import annotations

from typing import Any

_IMPORT_UNSET = object()
_psutil_module: Any = _IMPORT_UNSET
_torch_module: Any = _IMPORT_UNSET


class _LazyImportProxy:
    """Expose a patch-friendly module surface while deferring the real import."""

    def __init__(self, loader):
        self._loader = loader

    def _target(self) -> Any:
        return self._loader()

    def __getattr__(self, name: str) -> Any:
        target = self._target()
        if target is None:
            raise AttributeError(name)
        return getattr(target, name)

    def __bool__(self) -> bool:
        return self._target() is not None

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        target = self._target()
        if target is None:
            return "<lazy-missing-module>"
        return repr(target)


def _load_psutil_module() -> Any:
    global _psutil_module
    if _psutil_module is _IMPORT_UNSET:
        try:
            import psutil as _psutil
        except ImportError:
            _psutil_module = None
        else:
            _psutil_module = _psutil
    return None if _psutil_module is _IMPORT_UNSET else _psutil_module


def _load_torch_module() -> Any:
    global _torch_module
    if _torch_module is _IMPORT_UNSET:
        try:
            import torch as _torch
        except ImportError:
            _torch_module = None
        else:
            _torch_module = _torch
    return None if _torch_module is _IMPORT_UNSET else _torch_module


def get_psutil() -> Any:
    return psutil


def get_torch() -> Any:
    return torch


psutil: Any = _LazyImportProxy(_load_psutil_module)
torch: Any = _LazyImportProxy(_load_torch_module)


def reset_optional_runtime_caches() -> None:
    global _psutil_module, _torch_module
    if isinstance(psutil, _LazyImportProxy):
        _psutil_module = _IMPORT_UNSET
    if isinstance(torch, _LazyImportProxy):
        _torch_module = _IMPORT_UNSET


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
