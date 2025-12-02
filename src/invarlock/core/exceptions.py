from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InvarlockError(Exception):
    code: str
    message: str
    details: dict[str, Any] | None = None
    recoverable: bool = False

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[INVARLOCK:{self.code}] {self.message}"


class ConfigError(InvarlockError):
    """Configuration parsing/validation errors."""


class ValidationError(InvarlockError):
    """Domain validation errors for inputs/parameters."""


class DependencyError(InvarlockError):
    """Missing/invalid external dependency (package, binary, model file)."""


class ResourceError(InvarlockError):
    """Insufficient resources (CPU/GPU/RAM/Disk)."""


class TimeoutError(InvarlockError):
    """Operation timed out."""


class DataError(InvarlockError):
    """Dataset/provider errors (shape, availability, corruption)."""


class MetricsError(InvarlockError):
    """Metric computation errors (non-finite, mismatch)."""


class ModelLoadError(InvarlockError):
    """Model/weights loading failures."""


class AdapterError(InvarlockError):
    """Adapter-specific errors (resolution, device mapping)."""


class EditError(InvarlockError):
    """Model edit/transform failures."""


class GuardError(InvarlockError):
    """Guard setup/execution failures."""


class PolicyViolationError(InvarlockError):
    """Guard or policy violation (hard gate)."""


class PluginError(InvarlockError):
    """Plugin resolution/entry-point/import errors."""


class ObservabilityError(InvarlockError):
    """Observability/metrics/export issues."""


class VersionError(InvarlockError):
    """Version/ABI compatibility issues."""


__all__ = [
    "InvarlockError",
    "ConfigError",
    "ValidationError",
    "DependencyError",
    "ResourceError",
    "TimeoutError",
    "DataError",
    "MetricsError",
    "ModelLoadError",
    "AdapterError",
    "EditError",
    "GuardError",
    "PolicyViolationError",
    "PluginError",
    "ObservabilityError",
    "VersionError",
]
