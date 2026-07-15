from __future__ import annotations

from invarlock.reporting.validation.report import compute_validation_flags


def test_guard_metric_impact_missing_degradation_fails_closed() -> None:
    # Minimal ppl and other sections for validation function
    ppl = {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0}
    spectral = {"caps_applied": 0, "max_caps": 5}
    rmt = {"stable": True}
    invariants = {"status": "ok"}

    # A decision without its measured ratio cannot substantiate a pass.
    guard_metric_impact = {"degradation_limit": 0.01}

    flags = compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier="balanced",
        _ppl_metrics=None,
        target_ratio=None,
        guard_metric_impact=guard_metric_impact,
        primary_metric=None,
        moe=None,
        dataset_capacity=None,
    )

    assert flags["guard_metric_impact_acceptable"] is False
