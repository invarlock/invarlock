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
from .registry import PluginInfo, get_registry
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


# Lazy import CoreRunner to avoid importing heavy dependencies (e.g., NumPy)
# during lightweight operations such as registry inspection or CLI startup.
def __getattr__(name: str):  # pragma: no cover - simple lazy import shim
    if name == "CoreRunner":
        from .runner import CoreRunner as _CoreRunner

        return _CoreRunner
    raise AttributeError(name)
