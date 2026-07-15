from __future__ import annotations

import pytest

from invarlock.reporting.report_normalization import normalize_baseline
from invarlock.reporting.report_types import create_empty_report


def _canonical_baseline(
    *,
    kind: str = "ppl_causal",
    final: float | None = 12.0,
    preview: float | None = 12.0,
) -> dict:
    baseline = create_empty_report()
    baseline["meta"]["model_id"] = "baseline-model"
    baseline["meta"]["adapter"] = "hf_causal"
    baseline["edit"]["name"] = "noop"
    baseline["metrics"]["primary_metric"] = {
        "kind": kind,
        "final": final,
        "preview": preview,
    }
    return baseline


def test_normalize_baseline_derives_ppl_from_primary_metric() -> None:
    normalized = normalize_baseline(_canonical_baseline())

    assert normalized["model_id"] == "baseline-model"
    assert normalized["ppl_final"] == 12.0
    assert normalized["ppl_preview"] == 12.0


def test_normalize_baseline_rejects_missing_metric_despite_windows() -> None:
    baseline = _canonical_baseline(final=None, preview=None)
    baseline["evaluation_windows"] = {
        "preview": {"window_ids": ["p1"], "logloss": [1.0]},
        "final": {
            "window_ids": ["f1", "f2"],
            "logloss": [1.0, 3.0],
            "token_counts": [1, 3],
        },
    }

    with pytest.raises(ValueError, match="Invalid canonical RunReport structure"):
        normalize_baseline(baseline)


def test_normalize_baseline_rejects_missing_metric_with_zero_weight_windows() -> None:
    baseline = _canonical_baseline(final=None, preview=None)
    windows = {
        "window_ids": ["w1", "w2"],
        "logloss": [0.2, 0.4],
        "token_counts": [0, 0],
    }
    baseline["evaluation_windows"] = {"preview": windows, "final": windows}

    with pytest.raises(ValueError, match="Invalid canonical RunReport structure"):
        normalize_baseline(baseline)


def test_normalize_baseline_rejects_non_finite_ppl_evidence() -> None:
    baseline = _canonical_baseline(final=None, preview=None)
    baseline["evaluation_windows"] = {
        "preview": {"window_ids": ["p1"], "logloss": [float("inf")]},
        "final": {"window_ids": ["f1"], "logloss": [float("inf")]},
    }

    with pytest.raises(ValueError, match="Invalid canonical RunReport structure"):
        normalize_baseline(baseline)


def test_normalize_baseline_accepts_zero_accuracy_without_ppl_fields() -> None:
    normalized = normalize_baseline(
        _canonical_baseline(kind="accuracy", final=0.0, preview=0.0)
    )

    assert normalized["primary_metric"]["final"] == 0.0
    assert "ppl_final" not in normalized
    assert "ppl_preview" not in normalized


def test_normalize_baseline_rejects_legacy_summary_shape() -> None:
    legacy = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }

    with pytest.raises(ValueError, match="legacy baseline"):
        normalize_baseline(legacy)


def test_normalize_baseline_accepts_canonical_comparison_baseline() -> None:
    comparison_output = {
        "run_id": "r1",
        "model_id": "m",
        "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        "ppl_final": 10.0,
    }

    assert normalize_baseline(comparison_output) == comparison_output
