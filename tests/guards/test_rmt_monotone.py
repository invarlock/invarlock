import pytest

from invarlock.guards.rmt import rmt_growth_ratio, within_deadband


def test_rmt_growth_ratio_monotone() -> None:
    ratio = rmt_growth_ratio(1.05, 1.0, 1.0, 1.0)
    assert ratio == pytest.approx(1.05, rel=1e-6)

    # Guard against divide-by-zero in MP edges.
    ratio2 = rmt_growth_ratio(2.0, 0.0, 1.0, 0.0)
    assert ratio2 == pytest.approx(2.0, rel=1e-6)


def test_within_deadband_accepts_small_drift() -> None:
    assert within_deadband(1.08, 1.0, 0.1) is True
    assert within_deadband(1.1, 1.0, 0.1) is True
    assert within_deadband(1.2, 1.0, 0.1) is False
