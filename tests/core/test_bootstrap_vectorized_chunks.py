from __future__ import annotations

import numpy as np
import pytest

from invarlock.core import bootstrap
from invarlock.reporting import verify_bootstrap_math


def _scalar_resampled_means(
    values: np.ndarray,
    *,
    weights: np.ndarray | None,
    replicates: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    statistics = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = generator.integers(0, values.size, size=values.size)
        selected_values = values[selected]
        if weights is None:
            statistics[index] = float(np.mean(selected_values))
            continue
        selected_weights = weights[selected]
        total = float(selected_weights.sum())
        statistics[index] = (
            float(np.mean(selected_values))
            if total <= 0.0
            else float(np.dot(selected_values, selected_weights) / total)
        )
    return statistics


@pytest.mark.parametrize("weighted", [False, True])
def test_vectorized_bootstrap_chunks_preserve_scalar_rng_semantics(
    weighted: bool,
) -> None:
    values = np.asarray([0.2, 0.5, 0.9, 1.4, 1.8], dtype=float)
    weights = np.asarray([1.0, 0.0, 3.0, 7.0, 2.0]) if weighted else None
    expected = _scalar_resampled_means(
        values,
        weights=weights,
        replicates=257,
        seed=91,
    )

    observed = bootstrap._resampled_mean_statistics(  # noqa: SLF001
        values,
        weights=weights,
        replicates=257,
        rng=np.random.default_rng(91),
    )
    replayed = verify_bootstrap_math._resampled_means(  # noqa: SLF001
        values,
        weights=weights,
        replicates=257,
        generator=np.random.default_rng(91),
    )

    assert observed == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert replayed == pytest.approx(expected, rel=0.0, abs=1e-15)


def test_vectorized_bootstrap_is_chunk_boundary_invariant(monkeypatch) -> None:
    baseline = np.linspace(0.8, 1.4, 200)
    final = baseline + np.sin(np.arange(200)) * 0.01
    weights = np.arange(1, 201, dtype=float)

    monkeypatch.setattr(bootstrap, "_BOOTSTRAP_CHUNK_BYTES", 2048)
    producer = bootstrap.compute_paired_delta_log_ci(
        final,
        baseline,
        weights=weights,
        replicates=3200,
        seed=503,
    )
    monkeypatch.setattr(verify_bootstrap_math, "_BOOTSTRAP_CHUNK_BYTES", 2048)
    verifier = verify_bootstrap_math.replay_paired_delta_log_ci(
        final,
        baseline,
        weights=weights,
        replicates=3200,
        seed=503,
    )

    assert verifier == pytest.approx(producer, rel=0.0, abs=1e-15)
    assert (
        bootstrap._bootstrap_chunk_rows(  # noqa: SLF001
            200,
            3200,
            arrays_per_index=3,
        )
        < 3200
    )
