"""Optional observation-only numeric diagnostics for InvarLock.

NumPy is imported only when a diagnostic symbol is used, so the core package
can be imported and can verify or report evidence without the diagnostics
extra installed.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .observations import (
        DiagnosticInputError,
        canonical_observation_bytes,
        rmt_observation,
        spectral_observation,
        variance_observation,
    )

__all__ = [
    "DiagnosticInputError",
    "canonical_observation_bytes",
    "rmt_observation",
    "spectral_observation",
    "variance_observation",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    try:
        module = import_module("invarlock.diagnostics.observations")
    except ModuleNotFoundError as exc:
        if exc.name == "numpy" or exc.name == "numpy.typing":
            raise ImportError(
                "diagnostics require NumPy; install the optional "
                "'invarlock[diagnostics]' extra"
            ) from exc
        raise
    return getattr(module, name)
