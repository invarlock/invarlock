from __future__ import annotations

from invarlock.reporting.certificate import _compute_validation_flags


def test_validation_flags_hysteresis_applied_ratio_gate() -> None:
    # Ratio just above base limit but within hysteresis → accepted and hysteresis_applied
    flags = _compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.101, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        target_ratio=None,
        guard_overhead={},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 200000},
    )
    assert flags["primary_metric_acceptable"] is True
    assert flags.get("hysteresis_applied") is True


def test_validation_flags_sample_size_floor_blocks_acceptance() -> None:
    # Insufficient tokens → tokens_ok False → primary_metric_acceptable False
    flags = _compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 1000, "final_total_tokens": 1000},
        target_ratio=None,
        guard_overhead={},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 1_000_000},
    )
    assert flags["primary_metric_acceptable"] is False


def test_validation_flags_guard_overhead_variants() -> None:
    # Evaluated explicit pass
    f_pass = _compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        target_ratio=None,
        guard_overhead={"passed": True, "evaluated": True},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 200000},
    )
    assert f_pass["guard_overhead_acceptable"] is True

    # Evaluated explicit fail
    f_fail = _compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        target_ratio=None,
        guard_overhead={"passed": False, "evaluated": True},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 200000},
    )
    assert f_fail["guard_overhead_acceptable"] is False

    # Ratio non-finite → treated as pass
    f_nan = _compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        target_ratio=None,
        guard_overhead={"overhead_ratio": float("nan"), "overhead_threshold": 0.01},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 200000},
    )
    assert f_nan["guard_overhead_acceptable"] is True


def test_validation_flags_ratio_ci_upper_bound_gates() -> None:
    # Point passes, but CI upper bound exceeds → unacceptable
    flags = _compute_validation_flags(
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
        guard_overhead={},
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 200000},
    )
    assert flags["primary_metric_acceptable"] is False
