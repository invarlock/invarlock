from __future__ import annotations

import math

import pytest

from invarlock.eval.metrics import AccuracyMetric, PerplexityMetric


def test_perplexity_metric_handles_invalid_records() -> None:
    metric = PerplexityMetric()
    records = [
        None,
        {"logloss": "bad", "token_count": 1},
        {"logloss": 0.0, "token_count": 0},
        {"logloss": 0.0, "token_count": 1},
    ]
    assert metric.compute(None, records) == pytest.approx(1.0)

    empty = [{"logloss": "nan"}]
    assert math.isnan(metric.compute(None, empty))


def test_accuracy_metric_compute() -> None:
    metric = AccuracyMetric()
    records = [
        {"label": 1, "prediction": 1},
        {"label": 0, "prediction": 1},
        {"label": 0, "prediction": 0},
    ]
    assert metric.compute(None, records) == pytest.approx(2 / 3)
