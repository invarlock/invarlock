"""Exact contract for primary-metric tail gate outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_TAIL_MODES = frozenset({"off", "warn", "fail"})


class PrimaryMetricTailContractError(ValueError):
    """Raised when a primary-metric tail outcome is not exact."""


@dataclass(frozen=True)
class PrimaryMetricTailOutcome:
    """The three fields that determine the tail gate outcome."""

    mode: str
    evaluated: bool
    passed: bool

    @property
    def acceptable(self) -> bool:
        """Whether policy permits publication of this exact outcome."""

        return not (self.mode == "fail" and self.evaluated and not self.passed)


def require_primary_metric_tail(value: Any) -> PrimaryMetricTailOutcome:
    """Parse an exact tail outcome without truthiness or default coercions."""

    if type(value) is not dict:
        raise PrimaryMetricTailContractError(
            "primary_metric_tail must be a JSON object"
        )
    missing = [field for field in ("mode", "evaluated", "passed") if field not in value]
    if missing:
        raise PrimaryMetricTailContractError(
            "primary_metric_tail is missing required fields: " + ", ".join(missing)
        )
    mode = value["mode"]
    evaluated = value["evaluated"]
    passed = value["passed"]
    if type(mode) is not str or mode not in _TAIL_MODES:
        raise PrimaryMetricTailContractError(
            "primary_metric_tail.mode must be exactly off, warn, or fail"
        )
    if type(evaluated) is not bool:
        raise PrimaryMetricTailContractError(
            "primary_metric_tail.evaluated must be a boolean"
        )
    if type(passed) is not bool:
        raise PrimaryMetricTailContractError(
            "primary_metric_tail.passed must be a boolean"
        )
    return PrimaryMetricTailOutcome(mode=mode, evaluated=evaluated, passed=passed)


__all__ = [
    "PrimaryMetricTailContractError",
    "PrimaryMetricTailOutcome",
    "require_primary_metric_tail",
]
