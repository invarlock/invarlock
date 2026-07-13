from __future__ import annotations

import json

import numpy as np
import pytest

from invarlock.core.bootstrap import compute_paired_delta_log_ci
from invarlock.reporting.verify_bootstrap_math import replay_paired_delta_log_ci


@pytest.mark.parametrize("weighted", [False, True])
def test_verifier_bca_matches_producer_known_vectors(weighted: bool) -> None:
    baseline = [1.0, 1.2, 0.8, 1.4, 0.9, 1.1]
    subject = [1.02, 1.18, 0.82, 1.35, 0.93, 1.09]
    weights = [3, 5, 2, 7, 4, 6] if weighted else None
    expected = compute_paired_delta_log_ci(
        subject,
        baseline,
        weights=weights,
        method="bca",
        replicates=800,
        alpha=0.05,
        seed=503,
    )

    observed = replay_paired_delta_log_ci(
        subject,
        baseline,
        weights=weights,
        method="bca",
        replicates=800,
        alpha=0.05,
        seed=503,
    )

    assert observed == pytest.approx(expected, rel=0.0, abs=1e-15)
    golden = (
        (-0.03729679179667523, 0.011999999999999999)
        if weighted
        else (-0.02906584119793993, 0.018086808908856604)
    )
    assert observed == pytest.approx(golden, rel=0.0, abs=1e-15)


def test_verifier_bca_is_canonical_json_round_trip_invariant() -> None:
    payload = {
        "baseline": [1.0, 1.2, 0.8, 1.4],
        "subject": [1.02, 1.18, 0.82, 1.35],
        "weights": [3, 5, 2, 7],
    }
    parsed = json.loads(json.dumps(payload, sort_keys=True))

    before = replay_paired_delta_log_ci(
        payload["subject"],
        payload["baseline"],
        weights=payload["weights"],
        replicates=500,
        seed=503,
    )
    after = replay_paired_delta_log_ci(
        parsed["subject"],
        parsed["baseline"],
        weights=parsed["weights"],
        replicates=500,
        seed=503,
    )

    assert np.asarray(after) == pytest.approx(np.asarray(before), rel=0.0, abs=0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "percentile"}, "supports only bca"),
        ({"replicates": 0}, "replicates must be positive"),
        ({"alpha": 0.0}, "alpha must be between"),
        ({"alpha": 1.0}, "alpha must be between"),
    ],
)
def test_verifier_bca_rejects_unsupported_controls(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replay_paired_delta_log_ci([1.0], [1.0], **kwargs)


@pytest.mark.parametrize(
    ("final", "baseline", "weights", "message"),
    [
        ([], [], None, "non-empty one-dimensional"),
        ([[1.0]], [1.0], None, "one-dimensional"),
        ([float("nan")], [1.0], None, "must be finite"),
        ([1.0], [1.0], [], "weights length"),
        ([1.0], [1.0], [-1.0], "non-negative"),
        ([1.0], [1.0], [0.0], "positive sum"),
    ],
)
def test_verifier_bca_rejects_malformed_samples(
    final: list, baseline: list, weights: list | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replay_paired_delta_log_ci(final, baseline, weights=weights)


def test_verifier_bca_length_policy_and_degenerate_intervals() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        replay_paired_delta_log_ci([1.0, 2.0], [1.0])

    assert replay_paired_delta_log_ci([1.0, 2.0], [0.5], strict_lengths=False) == (
        0.5,
        0.5,
    )
    assert replay_paired_delta_log_ci([1.0, 2.0], [0.5, 1.5], weights=[1.0, 1.0]) == (
        0.5,
        0.5,
    )


def test_verifier_bca_weighted_resampling_handles_zero_weight_draws() -> None:
    observed = replay_paired_delta_log_ci(
        [1.0, 3.0, 9.0],
        [0.5, 2.0, 7.0],
        weights=[1.0, 0.0, 0.0],
        replicates=100,
        seed=7,
    )
    assert all(np.isfinite(observed))
    assert observed[0] <= observed[1]
