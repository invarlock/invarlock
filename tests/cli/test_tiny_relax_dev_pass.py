from invarlock.reporting.certificate import _compute_validation_flags


def test_tiny_relax_env_relaxes_gates(monkeypatch):
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    ppl = {"preview_final_ratio": 1.20, "ratio_vs_baseline": float("nan")}
    spectral = {"caps_applied": 0, "max_caps": 5}
    rmt = {"stable": True}
    invariants = {"status": "pass"}
    guard_overhead = {"overhead_ratio": float("nan"), "overhead_threshold": 0.01}
    flags = _compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 0, "final_total_tokens": 0},
        target_ratio=None,
        guard_overhead=guard_overhead,
        primary_metric=None,
        moe=None,
        dataset_capacity=None,
    )
    assert flags["preview_final_drift_acceptable"] is True
    assert flags["primary_metric_acceptable"] is True
    assert flags["guard_overhead_acceptable"] is True
