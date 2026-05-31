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


def resolve_command_exit_code(exc: Exception, *, profile: str | None) -> int:
    """Resolve shell exit codes deterministically from error class and profile."""

    prof = str(profile or "").strip().lower()

    if isinstance(exc, (ConfigError, ValidationError, DataError)):
        return 2
    if isinstance(exc, ValueError) and "invalid runreport" in str(exc).strip().lower():
        return 2
    if isinstance(exc, InvarlockError):
        return 3 if prof in {"ci", "ci_cpu", "release"} else 1
    return 1


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
    "resolve_command_exit_code",
]
