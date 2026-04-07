from __future__ import annotations

import math

import pytest

from invarlock.core.exceptions import ValidationError
from invarlock.eval import primary_metric as pm_mod
from invarlock.eval.primary_metric import (
    MetricContribution,
    _Accuracy,
    _coerce_float,
    _coerce_int,
    _PPLCausal,
    compute_primary_metric_from_report,
    get_metric,
    infer_binary_label_from_ids,
    validate_primary_metric_block,
)


def test_ppl_causal_point_from_windows_and_finalize() -> None:
    metric = _PPLCausal()
    windows = {"logloss": [1.0, 1.5], "token_counts": [4, 6]}
    ppl = metric.point_from_windows(windows=windows)
    assert ppl == pytest.approx(math.exp((1.0 * 4 + 1.5 * 6) / 10.0))

    metric.accumulate(MetricContribution(value=1.0, weight=4))
    metric.accumulate(MetricContribution(value=1.5, weight=6))
    assert metric.finalize() == pytest.approx(ppl)


def test_ppl_causal_paired_compare_uses_defaults() -> None:
    metric = _PPLCausal()
    subj = [MetricContribution(1.0, 1.0), MetricContribution(1.5, 1.0)]
    base = [MetricContribution(1.2, 1.0), MetricContribution(1.4, 1.0)]

    out = metric.paired_compare(subj, base, reps=None, seed=None, ci_level=None)
    assert out["kind"] == "ppl_causal"
    assert out["paired"] is True
    assert out["reps"] == metric.defaults.reps
    assert out["ci_level"] == metric.defaults.ci_level
    assert "delta" in out and "ci" in out and "display_ci" in out


def test_accuracy_point_from_windows_and_policies() -> None:
    metric = _Accuracy()
    # Per-example path: 2/3 correct
    win = {"example_correct": [1, 0, 1]}
    assert metric.point_from_windows(windows=win) == pytest.approx(2.0 / 3.0)

    # Aggregate path: abstain exclusion and ties handling
    win2 = {
        "correct_total": 8,
        "total": 12,
        "abstain_total": 2,
        "ties_total": 1,
        "policy": {
            "exclude_abstain": True,
            "ties_count_as_correct": True,
        },
    }
    acc = metric.point_from_windows(windows=win2)
    # total=12, abstain=2 -> 10; ties add 1 to correct → 9/10
    assert acc == pytest.approx(0.9)


def test_accuracy_point_from_windows_handles_bad_policy_safely() -> None:
    metric = _Accuracy()
    win = {
        "correct_total": 5,
        "total": 10,
        "policy": "not-a-dict",
        "abstain_total": "bad",
        "ties_total": "bad",
    }
    acc = metric.point_from_windows(windows=win)
    assert acc == pytest.approx(0.5)


def test_accuracy_point_from_windows_all_invalid_examples_returns_nan() -> None:
    metric = _Accuracy()
    acc = metric.point_from_windows(windows={"example_correct": [object(), "bad"]})
    assert math.isnan(acc)


def test_compute_primary_metric_from_report_empty_windows_returns_nan() -> None:
    payload = compute_primary_metric_from_report({}, kind="ppl_causal", baseline=None)
    assert math.isnan(payload["preview"])
    assert math.isnan(payload["final"])
    assert math.isnan(payload["ratio_vs_baseline"])


def test_validate_primary_metric_block_success_and_failure() -> None:
    block = {"preview": 1.0, "final": 2.0}
    assert validate_primary_metric_block(block) is block

    with pytest.raises(ValidationError):
        validate_primary_metric_block({"preview": "nan", "final": 2.0})


def test_validate_primary_metric_block_missing_preview_final_raises() -> None:
    with pytest.raises(ValidationError) as ei:
        validate_primary_metric_block({})
    assert getattr(ei.value, "code", None) == "E402"


def test_numeric_coercion_helpers_reject_bools_and_nonfinite_floats() -> None:
    assert _coerce_float(True) is None
    assert _coerce_int(False) is None
    assert _coerce_int(float("inf")) is None
    assert _coerce_int(float("nan")) is None


def test_ppl_causal_finalize_returns_nan_when_total_weight_non_positive() -> None:
    metric = _PPLCausal()
    metric._values = [1.0]  # type: ignore[attr-defined]
    metric._weights = [-1.0]  # type: ignore[attr-defined]
    assert math.isnan(metric.finalize())


def test_ppl_causal_paired_compare_uses_weight_fallback_when_needed(
    monkeypatch,
) -> None:
    metric = _PPLCausal()
    captured: dict[str, object] = {}

    def _fake_ci(subj_vals, base_vals, *, weights, **_kwargs):  # type: ignore[no-untyped-def]  # noqa: ARG001
        captured["weights"] = list(weights or [])
        return (0.0, 0.0)

    monkeypatch.setattr(
        "invarlock.eval.primary_metric.compute_paired_delta_log_ci", _fake_ci
    )
    metric.paired_compare(
        [MetricContribution(1.0, 0.0)],
        [MetricContribution(1.0, 0.0)],
        reps=1,
        seed=0,
        ci_level=0.95,
    )
    assert captured.get("weights") == [1.0]


def test_ppl_causal_accumulate_and_pairing_skip_invalid_contributions(
    monkeypatch,
) -> None:
    metric = _PPLCausal()
    metric.accumulate(MetricContribution(value="bad", weight=1.0))
    metric.accumulate(MetricContribution(value=0.4, weight="bad"))
    assert math.isnan(metric.finalize())

    monkeypatch.setattr(
        pm_mod,
        "compute_paired_delta_log_ci",
        lambda *args, **kwargs: (float("nan"), float("nan")),
    )
    out = metric.paired_compare(
        [MetricContribution(value="bad", weight=1.0), object()],
        [],
        reps=1,
        seed=0,
        ci_level=0.95,
    )
    assert math.isnan(out["subject_point"])
    assert math.isnan(out["baseline_point"])
    assert out["display"] == pytest.approx(1.0)


def test_get_metric_unknown_kind_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_metric("does-not-exist")


def test_infer_binary_label_from_ids_handles_negative_tokens() -> None:
    # Deterministic parity path should not fail on ints
    label = infer_binary_label_from_ids([-1, 2, 3])
    assert label in {0, 1}


def test_accuracy_accumulate_invalid_value_keeps_metric_empty() -> None:
    metric = _Accuracy()
    metric.accumulate(MetricContribution(value="bad"))
    assert math.isnan(metric.finalize())


def test_compute_primary_metric_accuracy_handles_non_dict_preview_window() -> None:
    report = {
        "evaluation_windows": {
            "preview": "bad-window",
            "final": {"input_ids": [[1, 2, 3], [4, 5, 6]]},
        }
    }
    payload = compute_primary_metric_from_report(report, kind="accuracy")
    assert math.isnan(payload["preview"])
    assert 0.0 <= payload["final"] <= 1.0


def test_accuracy_paired_compare_ignores_invalid_values() -> None:
    metric = _Accuracy()
    result = metric.paired_compare(
        [MetricContribution(value="bad"), {"value": True}, {"value": 1.0}],
        [MetricContribution(value="bad"), {"value": False}, {"value": 0.0}],
        reps=10,
        seed=0,
        ci_level=0.9,
    )

    assert result["subject_point"] == pytest.approx(1.0)
    assert result["baseline_point"] == pytest.approx(0.0)


def test_compute_primary_metric_accuracy_ensure_counts_handles_non_dict_windows() -> (
    None
):
    report = {
        "metrics": {
            "classification": {
                "preview": "bad-preview",
                "final": {"input_ids": [[1, 2, 3], "bad-seq"]},
            }
        }
    }
    baseline = {"metrics": {"primary_metric": {"kind": "vqa_accuracy", "final": 2.0}}}

    payload = compute_primary_metric_from_report(
        report, kind="accuracy", baseline=baseline
    )

    assert math.isnan(payload["preview"])
    assert payload["final"] == pytest.approx(1.0)
    assert math.isnan(payload["ratio_vs_baseline"])


def test_compute_accuracy_counts_prefers_explicit_bool_correct_flags() -> None:
    from invarlock.eval.primary_metric import compute_accuracy_counts

    correct, total = compute_accuracy_counts(
        [{"correct": True}, {"correct": False}, {"input_ids": [1, 2, 3]}]
    )

    assert (correct, total) == (2, 3)
