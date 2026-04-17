from __future__ import annotations

from enum import StrEnum


class ExecutionMode(StrEnum):
    CONTAINER = "container"
    TRUSTED_LOCAL = "trusted-local"


class RuntimeProvenanceMode(StrEnum):
    CONTAINER = "container"
    TRUSTED_LOCAL = "trusted-local"


__all__ = ["ExecutionMode", "RuntimeProvenanceMode"]
