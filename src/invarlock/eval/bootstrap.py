"""
Deterministic paired bootstrap helpers (eval layer).

These thin wrappers delegate to core bootstrap implementations and keep the
evaluation namespace stable for metric V1 consumers.
"""

from __future__ import annotations

from collections.abc import Iterable

from invarlock.core.bootstrap import compute_paired_delta_log_ci as _paired_delta_bca
from invarlock.core.exceptions import ValidationError


def paired_delta_mean_ci(
    subject: Iterable[float],
    baseline: Iterable[float],
    *,
    reps: int = 2000,
    seed: int = 0,
    ci_level: float = 0.95,
    method: str = "bca",
) -> tuple[float, float]:
    """
    Paired bootstrap CI for the mean delta of paired samples.

    Notes:
    - When `method == 'bca'`, this dispatches to the core BCa implementation.
    - `weights` are currently not supported; pass pre-aggregated per-example values.
    """
    alpha = 1.0 - float(ci_level)
    if method not in {"bca", "percentile"}:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "method must be 'bca' or 'percentile'"},
        )
    # The core function operates on log-loss deltas but is generic over the statistic
    # we pass in; here we reuse it for generic deltas by feeding raw values.
    # This is acceptable because the core function computes paired delta and bootsraps
    # the mean of the delta array regardless of semantic units.
    return _paired_delta_bca(
        list(subject),
        list(baseline),
        method="bca" if method == "bca" else "percentile",
        replicates=int(reps),
        alpha=alpha,
        seed=int(seed),
    )
