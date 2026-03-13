from __future__ import annotations

from typing import Any, TypedDict


class SpectralPolicy(TypedDict, total=False):
    """Type definition for spectral guard policy configuration."""

    sigma_quantile: float
    deadband: float
    scope: str
    estimator: dict[str, Any]
    degeneracy: dict[str, Any]
    correction_enabled: bool
    family_caps: dict[str, dict[str, float]]
    ignore_preview_inflation: bool
    max_caps: int
    multiple_testing: dict[str, Any]


__all__ = ["SpectralPolicy"]
