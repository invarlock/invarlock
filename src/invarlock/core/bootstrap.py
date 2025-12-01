"""
InvarLock Core Bootstrap Utilities
==============================

Numerically stable bootstrap helpers for evaluation metrics.

This module provides bias-corrected and accelerated (BCa) confidence
intervals tailored for paired log-loss statistics used by the runner
and safety certificate reports.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from statistics import NormalDist

import numpy as np

__all__ = [
    "compute_logloss_ci",
    "compute_paired_delta_log_ci",
    "logspace_to_ratio_ci",
]


Normal = NormalDist()


def _ensure_array(samples: Iterable[float]) -> np.ndarray:
    """Coerce iterable of floats to a 1-D NumPy array."""
    arr = np.asarray(list(samples), dtype=float)
    if arr.ndim != 1:
        raise ValueError("samples must be 1-dimensional")
    if arr.size == 0:
        raise ValueError("samples cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("samples must be finite")
    return arr


def _percentile_interval(stats: np.ndarray, alpha: float) -> tuple[float, float]:
    """Return lower/upper bounds from an array of bootstrap statistics."""
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(stats, lower_q)), float(np.percentile(stats, upper_q))


def _bca_interval(
    samples: np.ndarray,
    *,
    stat_fn: Callable[[np.ndarray], float],
    replicates: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Compute a BCa interval for the given statistic.

    Based on Efron & Tibshirani (1994). Handles small-sample edge cases by
    falling back to percentile intervals when the acceleration term cannot
    be computed (e.g., duplicate samples).
    """
    n = samples.size
    if n < 2:
        stat = stat_fn(samples)
        return float(stat), float(stat)

    # Bootstrap replicates of the statistic
    stats = np.empty(replicates, dtype=float)
    for i in range(replicates):
        idx = rng.integers(0, n, size=n)
        stats[i] = stat_fn(samples[idx])

    stats.sort()
    stat_hat = stat_fn(samples)

    # Bias-correction
    prop = np.clip((stats < stat_hat).mean(), 1e-6, 1.0 - 1e-6)
    z0 = Normal.inv_cdf(prop)

    # Jackknife estimates for acceleration
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack_sample = np.delete(samples, i)
        jack[i] = stat_fn(jack_sample)

    jack_mean = jack.mean()
    numerator = np.sum((jack_mean - jack) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    if denominator == 0.0:
        # Degenerate case → revert to percentile interval
        return _percentile_interval(stats, alpha)

    acc = numerator / denominator

    def _adjust_quantile(z_alpha: float) -> float:
        adj = z0 + (z0 + z_alpha) / max(1.0 - acc * (z0 + z_alpha), 1e-12)
        return float(Normal.cdf(adj))

    lower_pct = _adjust_quantile(Normal.inv_cdf(alpha / 2.0))
    upper_pct = _adjust_quantile(Normal.inv_cdf(1.0 - alpha / 2.0))

    return float(np.quantile(stats, lower_pct)), float(np.quantile(stats, upper_pct))


def _bootstrap_interval(
    samples: np.ndarray,
    *,
    stat_fn: Callable[[np.ndarray], float],
    method: str,
    replicates: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Dispatch helper supporting percentile and BCa intervals."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random.default_rng(seed)
    if method == "percentile":
        stats = np.empty(replicates, dtype=float)
        n = samples.size
        for i in range(replicates):
            idx = rng.integers(0, n, size=n)
            stats[i] = stat_fn(samples[idx])
        stats.sort()
        return _percentile_interval(stats, alpha)
    if method == "bca":
        return _bca_interval(
            samples,
            stat_fn=stat_fn,
            replicates=replicates,
            alpha=alpha,
            rng=rng,
        )

    raise ValueError(f"Unsupported bootstrap method '{method}'")


def compute_logloss_ci(
    logloss_samples: Iterable[float],
    *,
    method: str = "bca",
    replicates: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Compute a confidence interval over mean log-loss.

    Returns (lo, hi) in log-loss space.
    """
    samples = _ensure_array(logloss_samples)

    def stat_fn(data: np.ndarray) -> float:
        return float(np.mean(data))

    return _bootstrap_interval(
        samples,
        stat_fn=stat_fn,
        method=method,
        replicates=replicates,
        alpha=alpha,
        seed=seed,
    )


def compute_paired_delta_log_ci(
    final_logloss: Iterable[float],
    baseline_logloss: Iterable[float],
    *,
    method: str = "bca",
    replicates: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Compute a confidence interval over the paired mean delta of log-loss.

    This implementation uses simple mean, which equals the token-weighted mean
    when all evaluation windows have equal token counts. The runner enforces
    `seq_len == stride` (non-overlapping windows) and `window_match_fraction == 1.0`
    (perfect pairing), so the equal-weight simplification applies. See
    docs/assurance/01-eval-math-proof.md for the full derivation.

    Args:
        final_logloss: Iterable of per-window log-loss values after the edit/guard.
        baseline_logloss: Iterable of paired per-window log-loss values (before edit).

    Returns:
        (lo, hi) bounds of Δlog-loss such that ratio CI = exp(bounds).
    """
    final_arr = _ensure_array(final_logloss)
    base_arr = _ensure_array(baseline_logloss)
    if final_arr.size != base_arr.size:
        size = min(final_arr.size, base_arr.size)
        final_arr = final_arr[:size]
        base_arr = base_arr[:size]
    if final_arr.size == 0:
        return 0.0, 0.0

    delta = final_arr - base_arr
    if np.allclose(delta, delta[0]):
        mean_delta = float(delta.mean())
        return mean_delta, mean_delta

    def stat_fn(data: np.ndarray) -> float:
        return float(np.mean(data))

    return _bootstrap_interval(
        delta,
        stat_fn=stat_fn,
        method=method,
        replicates=replicates,
        alpha=alpha,
        seed=seed,
    )


def logspace_to_ratio_ci(delta_log_ci: tuple[float, float]) -> tuple[float, float]:
    """Convert Δlog-loss bounds to ratio (perplexity) space."""
    lo, hi = delta_log_ci
    return math.exp(lo), math.exp(hi)
