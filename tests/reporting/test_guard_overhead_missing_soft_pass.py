from __future__ import annotations

from invarlock.reporting.certificate import _compute_validation_flags


def test_guard_overhead_missing_ratio_treated_as_pass() -> None:
    # Minimal ppl and other sections for validation function
    ppl = {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0}
    spectral = {"caps_applied": 0, "max_caps": 5}
    rmt = {"stable": True}
    invariants = {"status": "ok"}

    # Guard overhead payload missing ratio should be treated as pass
    guard_overhead = {"overhead_threshold": 0.01}

    flags = _compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier="balanced",
        _ppl_metrics=None,
        target_ratio=None,
        guard_overhead=guard_overhead,
        primary_metric=None,
        moe=None,
        dataset_capacity=None,
    )

    assert flags["guard_overhead_acceptable"] is True
