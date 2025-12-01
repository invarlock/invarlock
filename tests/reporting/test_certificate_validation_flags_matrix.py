from __future__ import annotations

import pytest

from invarlock.reporting.certificate import (
    _compute_quality_overhead_from_guard,
    _compute_validation_flags,
)


def test_accuracy_min_examples_fraction_precedence() -> None:
    # accuracy path: enforce min_examples_fraction over n_final
    pm = {"kind": "accuracy", "ratio_vs_baseline": 0.5, "n_final": 100}
    flags = _compute_validation_flags(
        ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        primary_metric=pm,
        dataset_capacity={"examples_available": 300},
    )
    # Balanced min_examples=200 and min_examples_fraction=0.01→ 200 floor; n_final=100 → not acceptable
    assert flags["primary_metric_acceptable"] is False


def test_tiny_relax_env_widens_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    # In tiny relax, undefined ratio becomes acceptable and tokens floors relax
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    try:
        flags = _compute_validation_flags(
            ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": float("nan")},
            spectral={},
            rmt={},
            invariants={},
            tier="conservative",
            _ppl_metrics={"preview_total_tokens": 0, "final_total_tokens": 0},
        )
        assert flags["primary_metric_acceptable"] is True
    finally:
        monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)


def test_quality_overhead_from_guard_accuracy_delta_pp() -> None:
    # Construct bare/guarded with higher-is-better metric to exercise delta_pp path
    bare = {
        "metrics": {
            "primary_metric": {"kind": "accuracy", "final": 0.70},
            "classification": {"final": {"correct_total": 70, "total": 100}},
        }
    }
    guarded = {
        "metrics": {
            "primary_metric": {"kind": "accuracy", "final": 0.73},
            "classification": {"final": {"correct_total": 73, "total": 100}},
        }
    }
    qo = _compute_quality_overhead_from_guard(
        {"bare_report": bare, "guarded_report": guarded}, pm_kind_hint="accuracy"
    )
    assert qo and qo.get("basis") == "delta_pp" and abs(qo.get("value") - 3.0) < 1e-6
