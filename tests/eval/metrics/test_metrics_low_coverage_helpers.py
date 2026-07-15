from __future__ import annotations

import math

import pytest

import invarlock.eval.bench_policy as bench
from invarlock.core.bootstrap import paired_delta_mean_ci
from invarlock.core.exceptions import ValidationError
from invarlock.eval.tasks import (
    _lcs_len,
    _rouge_l,
    accuracy_from_records,
    bleu1_from_records,
    exact_match_from_records,
    rouge_l_from_records,
)
from invarlock.guards.spectral_detection import (
    classify_model_families,
    compute_z_scores,
)
from invarlock.guards.spectral_measurement import compute_spectral_norms


def test_paired_delta_mean_ci_supports_weights_and_percentile() -> None:
    lo, hi = paired_delta_mean_ci(
        [1.0, 2.0, 4.0],
        [0.5, 2.5, 3.0],
        weights=[1.0, 2.0, 4.0],
        reps=50,
        seed=7,
        method="percentile",
        ci_level=0.9,
    )
    assert math.isfinite(lo)
    assert math.isfinite(hi)
    assert lo <= hi


def test_paired_delta_mean_ci_rejects_bad_method() -> None:
    with pytest.raises(ValidationError):
        paired_delta_mean_ci([1.0, 2.0], [1.0, 2.0], method="bogus")


def test_bench_golden_constants_are_exported() -> None:
    assert bench.BENCH_GOLDEN_ID == "bench-golden-2025-12-13"
    assert len(bench.BENCH_GOLDEN_SHA256) == 64
    assert "BENCH_GOLDEN_ID" in bench.__all__
    assert "BENCH_GOLDEN_SHA256" in bench.__all__


def test_spectral_owned_modules_export_expected_symbols() -> None:
    assert callable(compute_z_scores)
    assert callable(compute_spectral_norms)
    assert callable(classify_model_families)


def test_accuracy_from_records_covers_variants_and_nan() -> None:
    assert math.isnan(accuracy_from_records(["skip", {"missing": "fields"}]))

    records = [
        {"correct": True},
        {"label": 1, "prediction": 1},
        {"labels": [1, 0], "predictions": [1, 1]},
        {"label": False, "prediction": False},
        {"labels": [True, False], "predictions": [True]},
        {"label": "x", "pred": "x"},
        {"label": "y", "predictions": "z"},
        {"labels": "left", "pred": "left"},
        {"label": None, "prediction": "skip"},
    ]
    assert accuracy_from_records(records) == pytest.approx(7 / 9)


def test_exact_match_from_records_covers_answer_shapes_and_nan() -> None:
    assert math.isnan(exact_match_from_records([{"prediction": "Paris"}]))

    records = [
        {"prediction": "  Paris  ", "answers": ["paris", "lyon"]},
        {"prediction": "tokyo", "answer": "Tokyo"},
        {"prediction": "rome", "answers": ["berlin", None]},
        {"prediction": "skip", "answers": None},
        {"prediction": None, "answer": "ignored"},
        "not-a-dict",
    ]
    assert exact_match_from_records(records) == pytest.approx(2 / 3)


def test_text_generation_helpers_cover_edge_paths_and_nan() -> None:
    assert math.isnan(bleu1_from_records([{"reference": "missing prediction"}]))
    assert math.isnan(rouge_l_from_records([{"reference": "missing prediction"}]))
    assert math.isnan(bleu1_from_records([{"prediction": "x"}]))
    assert math.isnan(rouge_l_from_records([{"prediction": "x"}]))
    assert _lcs_len([], ["a"]) == 0
    assert _rouge_l("left", "right") == 0.0

    records = [
        {"prediction": "the cat sat", "references": ["the cat sat"]},
        {"prediction": "tiny output", "reference": "tiny output with extras"},
        {"prediction": "ignored", "references": [None]},
        {"prediction": "", "reference": "nonempty"},
        "skip-me",
    ]

    bleu = bleu1_from_records(records)
    rouge = rouge_l_from_records(records)
    assert 0.0 < bleu <= 1.0
    assert 0.0 < rouge <= 1.0
