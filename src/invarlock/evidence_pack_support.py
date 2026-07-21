"""Shared result types for canonical evidence-pack verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class EvidencePackStatus(IntEnum):
    """Stable process statuses for evaluate/verify/report callers."""

    OK = 0
    USAGE = 2
    MISSING = 3
    FORMAT = 4
    SIGNATURE = 5
    INTEGRITY = 6
    REPORTS = 7
    INTEGRITY_ONLY = 8


@dataclass(frozen=True)
class EvidencePackResult:
    """Machine-readable verification payload and its process status."""

    payload: dict[str, Any]
    status: EvidencePackStatus
    manifest_digest: str | None = None


__all__ = ["EvidencePackResult", "EvidencePackStatus"]
