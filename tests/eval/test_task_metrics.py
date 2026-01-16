from __future__ import annotations

import math

import pytest

from invarlock.eval.metrics import AccuracyMetric, PerplexityMetric
from invarlock.eval.tasks import (
    accuracy_from_records,
    bleu1_from_records,
    exact_match_from_records,
    rouge_l_from_records,
)


def test_accuracy_from_records_simple() -> None:
    records = [
        {"label": "yes", "prediction": "yes"},
        {"label": "no", "prediction": "yes"},
    ]
    assert accuracy_from_records(records) == 0.5


def test_exact_match_from_records_variants() -> None:
    records = [
        {"prediction": "Paris", "answers": ["Paris", "Lyon"]},
        {"prediction": "tokyo", "answer": "Tokyo"},
        {"prediction": "madrid", "answers": ["rome", "berlin"]},
    ]
    assert exact_match_from_records(records) == pytest.approx(2 / 3)


def test_bleu_and_rouge_l_from_records() -> None:
    records = [
        {"prediction": "the cat sat", "references": ["the cat sat"]},
        {"prediction": "quick brown", "reference": "quick fox"},
    ]
    bleu = bleu1_from_records(records)
    rouge = rouge_l_from_records(records)
    assert 0.0 <= bleu <= 1.0
    assert 0.0 <= rouge <= 1.0
    assert bleu > 0.0
    assert rouge > 0.0


def test_metric_classes_compute() -> None:
    ppl_metric = PerplexityMetric()
    records = [
        {"logloss": 0.0, "token_count": 1},
        {"logloss": math.log(4.0), "token_count": 1},
    ]
    assert ppl_metric.compute(None, records) == pytest.approx(2.0)

    acc_metric = AccuracyMetric()
    acc_records = [
        {"label": 1, "prediction": 1},
        {"label": 0, "prediction": 1},
        {"label": 0, "prediction": 0},
    ]
    assert acc_metric.compute(None, acc_records) == pytest.approx(2 / 3)
