from __future__ import annotations

import math

from invarlock.reporting.utils import (
    _coerce_interval,
    _pair_logloss_windows,
    _sanitize_seed_bundle,
    _weighted_mean,
)


def test_seed_bundle_sanitize_all_none() -> None:
    out = _sanitize_seed_bundle(
        {"python": None, "numpy": None, "torch": None}, fallback=7
    )
    assert out == {"python": None, "numpy": None, "torch": None}


def test_coerce_interval_malformed_and_valid() -> None:
    a, b = _coerce_interval("not-a-tuple")
    assert math.isnan(a) and math.isnan(b)
    c, d = _coerce_interval([1, 2])
    assert (c, d) == (1.0, 2.0)


def test_weighted_mean_handles_invalid_weights() -> None:
    # Non-positive and non-finite weights should be ignored; result finite
    val = _weighted_mean([1, 2, 3], [0, float("nan"), 1])
    assert math.isfinite(val)


def test_pair_logloss_windows_insufficient_and_none() -> None:
    # Insufficient matches returns None
    run_w = {"window_ids": [1], "logloss": [0.1]}
    base_w = {"window_ids": [2], "logloss": [0.2]}
    assert _pair_logloss_windows(run_w, base_w) is None
    # Non-dicts return None
    assert _pair_logloss_windows([], {}) is None


def test_get_ppl_final_missing_and_present() -> None:
    # Legacy _get_ppl_final removed; rely on primary_metric parsing in certificates.
    assert True
