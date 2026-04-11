from __future__ import annotations

from enum import StrEnum


class AssuranceMode(StrEnum):
    ATTESTED = "attested"
    TRUSTED_LOCAL = "trusted-local"


__all__ = ["AssuranceMode"]
