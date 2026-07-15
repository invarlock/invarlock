from invarlock.reporting.validation.report import compute_validation_flags


def test_tiny_relax_argument_relaxes_gates():
    ppl = {"preview_final_ratio": 1.20, "ratio_vs_baseline": float("nan")}
    spectral = {"caps_applied": 0, "max_caps": 5}
    rmt = {"stable": True}
    invariants = {"status": "pass"}
    guard_metric_impact = {"degradation": float("nan"), "degradation_limit": 0.01}
    flags = compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 0, "final_total_tokens": 0},
        target_ratio=None,
        guard_metric_impact=guard_metric_impact,
        primary_metric=None,
        moe=None,
        dataset_capacity=None,
        tiny_relax=True,
    )
    assert flags["preview_final_drift_acceptable"] is True
    assert flags["primary_metric_acceptable"] is False
    assert flags["guard_metric_impact_acceptable"] is False
