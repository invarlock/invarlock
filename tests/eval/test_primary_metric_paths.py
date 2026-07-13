from __future__ import annotations

import math

import pytest

from invarlock.eval.primary_metric import (
    compute_accuracy_counts,
    compute_primary_metric_from_report,
    infer_binary_label_from_ids,
)


def test_compute_primary_metric_accuracy_uses_only_percentage_point_delta():
    report = {
        "metrics": {
            "classification": {
                "preview": {"correct_total": 8, "total": 10},
                "final": {"correct_total": 18, "total": 20},
                "counts_source": "measured",
            }
        }
    }
    baseline = {
        "metrics": {"primary_metric": {"kind": "accuracy", "final": 0.8}},
    }
    payload = compute_primary_metric_from_report(
        report, kind="accuracy", baseline=baseline
    )
    assert payload["n_preview"] == 10 and payload["n_final"] == 20
    assert payload["counts_source"] == "measured"
    assert payload["estimated"] is False
    expected_delta_pp = 100.0 * (
        payload["final"] - baseline["metrics"]["primary_metric"]["final"]
    )
    assert payload["delta_vs_baseline_pp"] == pytest.approx(expected_delta_pp)
    assert "ratio_vs_baseline" not in payload


def test_compute_primary_metric_accuracy_ignores_bool_counts_and_baseline_final():
    report = {
        "metrics": {
            "classification": {
                "preview": {"correct_total": True, "total": 10},
                "final": {"correct_total": 18, "total": 20},
                "counts_source": "measured",
            }
        }
    }
    baseline = {
        "metrics": {"primary_metric": {"kind": "accuracy", "final": True}},
    }
    payload = compute_primary_metric_from_report(
        report, kind="accuracy", baseline=baseline
    )
    assert math.isnan(payload["preview"])
    assert payload["final"] == pytest.approx(0.9)
    assert "delta_vs_baseline_pp" not in payload
    assert "ratio_vs_baseline" not in payload


def test_compute_primary_metric_accuracy_rejects_input_only_windows():
    report = {
        "evaluation_windows": {
            "preview": {"input_ids": [[1, 2, 3], [4, 5, 6]]},
            "final": {"input_ids": [[7, 8, 9]]},
        }
    }
    payload = compute_primary_metric_from_report(report, kind="accuracy")
    assert math.isnan(payload["preview"])
    assert math.isnan(payload["final"])
    assert payload["invalid"] is True


def test_compute_primary_metric_ppl_ratio_vs_baseline():
    report = {
        "evaluation_windows": {
            "preview": {"logloss": [1.0], "token_counts": [4]},
            "final": {"logloss": [1.5], "token_counts": [5]},
        }
    }
    baseline = {
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": math.exp(1.2)}}
    }
    payload = compute_primary_metric_from_report(
        report, kind="ppl_causal", baseline=baseline
    )
    assert payload["ratio_vs_baseline"] == pytest.approx(
        payload["final"] / baseline["metrics"]["primary_metric"]["final"]
    )


def test_compute_primary_metric_ppl_ignores_bool_baseline_final():
    report = {
        "evaluation_windows": {
            "preview": {"logloss": [1.0], "token_counts": [4]},
            "final": {"logloss": [1.5], "token_counts": [5]},
        }
    }
    baseline = {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": True}}}
    payload = compute_primary_metric_from_report(
        report, kind="ppl_causal", baseline=baseline
    )
    assert payload["final"] == pytest.approx(math.exp(1.5))
    assert "ratio_vs_baseline" not in payload


def test_infer_binary_label_handles_bad_tokens():
    assert infer_binary_label_from_ids(["not-int"]) == 0


@pytest.mark.parametrize("token", [True, float("inf"), float("-inf"), float("nan")])
def test_infer_binary_label_rejects_non_integral_special_values(token):
    assert infer_binary_label_from_ids([token]) == 0


def test_compute_accuracy_counts_ignores_invalid_records():
    correct, total = compute_accuracy_counts(
        [{"input_ids": [1, 2]}, {"input_ids": []}, {"foo": "bar"}]
    )
    assert (correct, total) == (0, 0)
