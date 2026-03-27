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
    # Supporting services
    "EventLogger",
    "CheckpointManager",
]
