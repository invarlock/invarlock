"""Guard namespace (`invarlock.guards`) re-exporting built-in guards."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from invarlock.core.abi import INVARLOCK_CORE_ABI as INVARLOCK_CORE_ABI

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .invariants import InvariantsGuard
    from .rmt import RMTGuard
    from .spectral import SpectralGuard
    from .variance import VarianceGuard

__all__ = [
    "InvariantsGuard",
    "SpectralGuard",
    "VarianceGuard",
    "RMTGuard",
    "INVARLOCK_CORE_ABI",
]


_GUARD_EXPORTS = {
    "InvariantsGuard": (".invariants", "InvariantsGuard"),
    "SpectralGuard": (".spectral", "SpectralGuard"),
    "VarianceGuard": (".variance", "VarianceGuard"),
    "RMTGuard": (".rmt", "RMTGuard"),
}


def __getattr__(name: str) -> object:
    target = _GUARD_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    return getattr(import_module(module_name, __name__), attr_name)
