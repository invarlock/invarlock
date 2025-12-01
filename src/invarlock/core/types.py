"""
InvarLock Core Types
================

Core type definitions and enums used throughout InvarLock.
Torch-independent type system for cross-module compatibility.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple


class EditType(Enum):
    """Types of model edits supported by InvarLock."""

    QUANTIZATION = "quantization"
    SPARSITY = "sparsity"
    MIXED = "mixed"


class GuardType(Enum):
    """Types of safety guards available."""

    INVARIANTS = "invariants"
    SPECTRAL = "spectral"
    VARIANCE = "variance"
    RMT = "rmt"
    NOOP = "noop"


class RunStatus(Enum):
    """Execution status for pipeline runs."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"
    CANCELLED = "cancelled"


class LogLevel(Enum):
    """Logging levels for events."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ModelInfo:
    """Basic model information."""

    model_id: str
    architecture: str
    parameters: int
    device: str
    precision: str = "float32"


@dataclass
class EditInfo:
    """Information about an applied edit."""

    name: str
    type: EditType
    parameters: dict[str, Any]
    compression_ratio: float | None = None
    target_metrics: dict[str, float] | None = None


@dataclass
class GuardResult:
    """Result from a guard validation."""

    guard_name: str
    passed: bool
    score: float | None = None
    threshold: float | None = None
    message: str | None = None
    details: dict[str, Any] | None = None


class ValidationResult(NamedTuple):
    """Result from validation operations."""

    passed: bool
    score: float
    threshold: float
    message: str = ""


@dataclass
class GuardOutcome:
    """Result from a guard execution."""

    name: str
    passed: bool
    action: str = "none"
    violations: list[dict[str, Any]] | None = None
    metrics: dict[str, Any] | None = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.metrics is None:
            self.metrics = {}


@dataclass
class PolicyConfig:
    """Configuration for guard policies."""

    on_violation: str = "warn"
    guard_overrides: dict[str, str] | None = None
    enable_auto_rollback: bool = False

    def __post_init__(self):
        if self.guard_overrides is None:
            self.guard_overrides = {}

    def get_action_for_guard(self, guard_name: str, requested_action: str) -> str:
        """Get the action for a specific guard."""
        # Check if there's an override for this guard
        if self.guard_overrides and guard_name in self.guard_overrides:
            return self.guard_overrides[guard_name]

        # If requested action is not 'none', use it
        if requested_action != "none":
            return requested_action

        # Fall back to global default
        return self.on_violation


def get_worst_action(actions: list[str]) -> str:
    """Get the worst (most severe) action from a list."""
    action_priority = {"none": 0, "warn": 1, "rollback": 2, "abort": 3}

    if not actions:
        return "none"

    return max(actions, key=lambda action: action_priority.get(action, 0))


# Type aliases for clarity
DeviceSpec = str | Any  # Device specification
ConfigDict = dict[str, Any]  # Configuration dictionary
MetricsDict = dict[str, float | int | str | bool]  # Metrics
LayerIndex = int  # Layer index
HeadIndex = int  # Attention head index
