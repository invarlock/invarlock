from __future__ import annotations

import importlib
import io
import warnings
from contextlib import redirect_stderr, redirect_stdout

_SAFE_IMPORT_ERRORS = (ImportError, AttributeError, RuntimeError, OSError)


def _safe_import(module_name: str, attr: str | None = None) -> bool:
    """Return True when a backend module (and optional symbol) imports cleanly."""

    try:
        with (
            warnings.catch_warnings(),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            module = importlib.import_module(module_name)
        if attr is None:
            return True
        return getattr(module, attr, None) is not None
    except _SAFE_IMPORT_ERRORS:
        return False


def bitsandbytes_runtime_available() -> bool:
    """Return True when bitsandbytes is importable on this host."""

    return _safe_import("bitsandbytes")


__all__ = [
    "bitsandbytes_runtime_available",
]
