from __future__ import annotations

from enum import Enum


class AssuranceMode(str, Enum):
    ATTESTED = "attested"
    TRUSTED_LOCAL = "trusted-local"


__all__ = ["AssuranceMode"]
