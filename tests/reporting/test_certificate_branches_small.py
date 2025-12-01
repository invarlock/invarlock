from __future__ import annotations

from invarlock.reporting import certificate as C


def test_pair_logloss_windows_invalid_and_small_samples():
    # Invalid inputs
    assert C._pair_logloss_windows(None, None) is None  # type: ignore[arg-type]
    assert C._pair_logloss_windows({"window_ids": [1]}, {"window_ids": [1]}) is None
    # Valid structure but <2 pairs → None
    run = {"window_ids": [1], "logloss": [1.0]}
    base = {"window_ids": [1], "logloss": [1.0]}
    assert C._pair_logloss_windows(run, base) is None


def test_pair_logloss_windows_happy_path_two_pairs():
    run = {"window_ids": [1, 2], "logloss": [1.0, 2.0]}
    base = {"window_ids": [1, 2], "logloss": [1.1, 1.9]}
    out = C._pair_logloss_windows(run, base)
    assert isinstance(out, tuple) and len(out) == 2
    pr, pb = out
    assert pr == [1.0, 2.0] and pb == [1.1, 1.9]


def test_compute_variance_policy_digest_variants():
    assert C._compute_variance_policy_digest({}) == ""
    h = C._compute_variance_policy_digest({"deadband": 0.02, "min_abs_adjust": 0.01})
    assert isinstance(h, str) and len(h) == 16


def test_thresholds_payload_and_hash_shape():
    payload = C._compute_thresholds_payload("balanced", resolved_policy={})
    assert set(payload).issuperset({"tier", "pm_ratio", "accuracy", "variance"})
    digest = C._compute_thresholds_hash(payload)
    assert isinstance(digest, str) and len(digest) == 16
