from __future__ import annotations

import pytest

from invarlock.primary_metric_tail import (
    PrimaryMetricTailContractError,
    require_primary_metric_tail,
)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"mode": "fail", "evaluated": True},
        {"mode": "FAIL", "evaluated": True, "passed": True},
        {"mode": "fail", "evaluated": 1, "passed": True},
        {"mode": "fail", "evaluated": True, "passed": 1},
        {"mode": "fail", "evaluated": True, "passed": "yes"},
    ],
)
def test_primary_metric_tail_rejects_coercible_or_incomplete_shapes(
    payload: object,
) -> None:
    with pytest.raises(PrimaryMetricTailContractError):
        require_primary_metric_tail(payload)


def test_primary_metric_tail_acceptability_is_policy_exact() -> None:
    assert require_primary_metric_tail(
        {"mode": "warn", "evaluated": True, "passed": False}
    ).acceptable
    assert require_primary_metric_tail(
        {"mode": "fail", "evaluated": False, "passed": False}
    ).acceptable
    assert not require_primary_metric_tail(
        {"mode": "fail", "evaluated": True, "passed": False}
    ).acceptable
