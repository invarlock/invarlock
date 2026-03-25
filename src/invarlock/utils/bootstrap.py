from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np

_FAST_MEAN_STATISTICS = {np.mean, np.nanmean}


def _is_mean_statistic(statistic: Callable[[Any], Any] | None) -> bool:
    if statistic is None or statistic in _FAST_MEAN_STATISTICS:
        return True
    name = getattr(statistic, "__name__", "")
    return bool(name in {"mean", "nanmean"})


def bootstrap_mean_statistics(
    data: np.ndarray,
    *,
    n_bootstrap: int,
    random_state: np.random.Generator,
    max_resample_elements: int = 1_000_000,
) -> np.ndarray:
    """Return bootstrap resample means for a 1D array using chunked vectorization."""
    if data.ndim != 1:
        raise ValueError("bootstrap_mean_statistics requires 1D input")
    if n_bootstrap <= 0:
        return np.empty(0, dtype=float)

    data = np.asarray(data, dtype=float)
    sample_size = int(data.size)
    if sample_size <= 0:
        return np.empty(0, dtype=float)

    chunk_rows = max(1, int(max_resample_elements) // max(sample_size, 1))
    chunk_rows = min(chunk_rows, int(n_bootstrap))

    stats = np.empty(int(n_bootstrap), dtype=float)
    for start in range(0, int(n_bootstrap), chunk_rows):
        stop = min(start + chunk_rows, int(n_bootstrap))
        indices = random_state.integers(
            0, sample_size, size=(stop - start, sample_size)
        )
        stats[start:stop] = data[indices].mean(axis=1, dtype=float)
    return stats


def bootstrap_statistics(
    data: np.ndarray,
    *,
    n_bootstrap: int,
    random_state: np.random.Generator,
    statistic: Callable[[Any], Any] | None = None,
) -> np.ndarray:
    """Return bootstrap statistics for a 1D array with a fast path for sample means."""
    data = np.asarray(data, dtype=float)
    if _is_mean_statistic(statistic):
        return bootstrap_mean_statistics(
            data,
            n_bootstrap=int(n_bootstrap),
            random_state=random_state,
        )

    stats = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        sample_idx = random_state.integers(0, data.size, size=data.size)
        stats[index] = float(cast(Callable[[Any], Any], statistic)(data[sample_idx]))
    return stats


def percentile_interval_from_statistics(
    statistics: np.ndarray, *, alpha: float
) -> tuple[float, float]:
    """Return the two-sided percentile interval for bootstrap statistics."""
    lower = float(np.percentile(statistics, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(statistics, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper
