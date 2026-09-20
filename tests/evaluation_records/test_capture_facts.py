"""Supplied metrics remain observations and cannot overwrite native values."""

import pytest

from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError
from invarlock.evaluation_records.capture_facts import merge_capture_scores


def test_named_observations_merge_without_mutation():
    native = {"quality": 0.5}
    assert merge_capture_scores(native, {}) == native
    assert merge_capture_scores(
        native, {"invarlock_scores": {"quality": 0.5, "latency_ms": 30}}
    ) == {"quality": 0.5, "latency_ms": 30}
    assert native == {"quality": 0.5}


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"quality": 0.6},
        {"": 1},
        {1: 1},
        {"x": True},
        {"x": "1"},
        {"x": float("nan")},
        {"x": float("inf")},
    ],
)
def test_invalid_or_conflicting_observations_fail(value):
    with pytest.raises(EvaluationRecordsError):
        merge_capture_scores({"quality": 0.5}, {"invarlock_scores": value})
