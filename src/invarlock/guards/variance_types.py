from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationBatchContext:
    window_ids: list[str]
    count: int
    observed_digest: str | None
    expected_digest: str | None = None


@dataclass(frozen=True)
class ScaleComputationResult:
    raw_scales: dict[str, float]
    filtered_scales: dict[str, float]
    backstop_used: bool
    trimmed_to_limit: bool


__all__ = ["CalibrationBatchContext", "ScaleComputationResult"]
