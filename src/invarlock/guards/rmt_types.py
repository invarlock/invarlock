from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


@dataclass
class RMTPolicy:
    """RMT guard policy configuration."""

    q: float | Literal["auto"] = "auto"
    deadband: float = 0.10
    margin: float = 1.5
    correct: bool = True


class RMTPolicyDict(TypedDict, total=False):
    """TypedDict version of the RMT guard policy."""

    q: float | Literal["auto"]
    deadband: float
    margin: float
    correct: bool
    epsilon_default: float
    epsilon_by_family: dict[str, float]
    activation_required: bool
    estimator: dict[str, Any]
    activation: dict[str, Any]
