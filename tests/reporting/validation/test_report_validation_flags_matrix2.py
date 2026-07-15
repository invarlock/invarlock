from __future__ import annotations

from invarlock.reporting.validation.report import compute_validation_flags
from tests.reporting._support_guard_metric_impact import canonical_ppl_impact


def test_validation_flags_hysteresis_applied_ratio_gate() -> None:
    # Ratio just above base limit but within hysteresis → accepted and hysteresis_applied
    flags = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.101, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        target_ratio=None,
        guard_metric_impact={},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 200000},
    )
    assert flags["primary_metric_acceptable"] is True
    assert flags.get("hysteresis_applied") is True


def test_validation_flags_sample_size_floor_blocks_acceptance() -> None:
    # Insufficient tokens → tokens_ok False → primary_metric_acceptable False
    flags = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 1000, "final_total_tokens": 1000},
        target_ratio=None,
        guard_metric_impact={},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 1_000_000},
    )
    assert flags["primary_metric_acceptable"] is False


def test_validation_flags_guard_metric_impact_variants() -> None:
    # Evaluated explicit pass
    f_pass = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        target_ratio=None,
        guard_metric_impact=canonical_ppl_impact(),
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 200000},
    )
    assert f_pass["guard_metric_impact_acceptable"] is True

    # Evaluated explicit fail
    f_fail = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        target_ratio=None,
        guard_metric_impact={"passed": False, "evaluated": True},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 200000},
    )
    assert f_fail["guard_metric_impact_acceptable"] is False

    # A non-finite ratio cannot substantiate a pass.
    f_nan = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        target_ratio=None,
        guard_metric_impact={"degradation": float("nan"), "degradation_limit": 0.01},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 200000},
    )
    assert f_nan["guard_metric_impact_acceptable"] is False


def test_validation_flags_ratio_ci_upper_bound_gates() -> None:
    # Point passes, but CI upper bound exceeds → unacceptable
    flags = compute_validation_flags(
        ppl={
            "ratio_vs_baseline": 1.08,
            "ratio_ci": (1.02, 1.12),
            "preview_final_ratio": 1.0,
        },
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        target_ratio=None,
        guard_metric_impact={},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 200000},
    )
    assert flags["primary_metric_acceptable"] is False


def test_ppl_reconcile_cannot_overwrite_ratio_ci_failure() -> None:
    flags = compute_validation_flags(
        ppl={
            "ratio_vs_baseline": 1.02,
            "ratio_ci": (0.99, 1.12),
            "preview_final_ratio": 1.0,
        },
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.02},
        dataset_capacity={"tokens_available": 200000},
    )

    assert flags["primary_metric_acceptable"] is False


def test_ppl_reconcile_cannot_overwrite_acceptance_lower_bound_failure() -> None:
    flags = compute_validation_flags(
        ppl={"ratio_vs_baseline": 0.90, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 0.90},
        pm_acceptance_range={"min": 0.95, "max": 1.10},
        dataset_capacity={"tokens_available": 200000},
    )

    assert flags["primary_metric_acceptable"] is False
