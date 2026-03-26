"""
InvarLock Core Module
=================

Core torch-independent interfaces and coordination logic.

This module provides the foundational abstractions and orchestration
for the InvarLock framework without requiring heavy dependencies.
"""

from .abi import INVARLOCK_CORE_ABI
from .api import Guard, ModelAdapter, ModelEdit, RunConfig, RunReport
from .checkpoint import CheckpointManager
from .events import EventLogger
from .exceptions import InvarlockError
from .types import (
    EditInfo,
    EditType,
    GuardResult,
    GuardType,
    LogLevel,
    ModelInfo,
    RunStatus,
)

__all__ = [
    # Core interfaces
    "ModelAdapter",
    "ModelEdit",
    "Guard",
    # ABI contract
    "INVARLOCK_CORE_ABI",
    "RunConfig",
    "RunReport",
    # Exceptions
    "InvarlockError",
    # Types and enums
    "EditType",
    "GuardType",
    "RunStatus",
    "LogLevel",
    "ModelInfo",
    "EditInfo",
    "GuardResult",
    # Registry and discovery
    "get_registry",
    "PluginInfo",
    # Supporting services
    "EventLogger",
    "CheckpointManager",
]


# Lazy imports avoid pulling in registry/runtime surfaces during lightweight
# CLI startup and helper imports.
def __getattr__(name: str):  # pragma: no cover - simple lazy import shim
    if name == "CoreRunner":
        from .runner import CoreRunner as _CoreRunner

        return _CoreRunner
    if name in {"PluginInfo", "get_registry"}:
        from .registry import PluginInfo as _PluginInfo
        from .registry import get_registry as _get_registry

        if name == "PluginInfo":
            return _PluginInfo
        return _get_registry
    raise AttributeError(name)
