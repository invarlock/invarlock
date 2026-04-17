from __future__ import annotations

from enum import StrEnum


class ExecutionMode(StrEnum):
    CONTAINER = "container"
    HOST = "host"


class RuntimeProvenanceMode(StrEnum):
    CONTAINER = "container"
    HOST = "host"


__all__ = ["ExecutionMode", "RuntimeProvenanceMode"]
