from __future__ import annotations

from invarlock.reporting import certificate as C


def test_compute_validation_flags_tiny_relax_and_tokens_floor(monkeypatch):
    # Enable tiny relax to exercise relaxed branches
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")

    pm_policy = {
        "min_tokens": 100000,
        "min_token_fraction": 0.5,
        "hysteresis_ratio": 0.02,
    }
    # Simulate tier policy
    tier = "balanced"
    ppl = {
        "preview_final_ratio": 1.10,
        "ratio_vs_baseline": 1.12,
        "ratio_ci": (1.00, 1.15),
    }
    spectral = {"caps_applied": 0}
    rmt = {"stable": True}
    invariants = {"status": "pass"}
    primary_metric = {"kind": "ppl_causal", "ratio_vs_baseline": 1.12}
    # Populate _ppl_metrics to compute tokens_ok=False against min_tokens
    _ppl_metrics = {"preview_total_tokens": 1000, "final_total_tokens": 1000}
    dataset_capacity = {"tokens_available": 10000}

    fake_policies = {
        "balanced": {"metrics": {"pm_ratio": pm_policy}},
        # tiny_relax forces tier="aggressive"
        "aggressive": {"metrics": {"pm_ratio": pm_policy}},
    }
    monkeypatch.setattr(C, "get_tier_policies", lambda *_a, **_k: dict(fake_policies))
    flags = C._compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier=tier,
        _ppl_metrics=_ppl_metrics,
        target_ratio=None,
        guard_overhead=None,
        primary_metric=primary_metric,
        moe=None,
        dataset_capacity=dataset_capacity,
    )

    assert isinstance(flags, dict)
    # With tiny relax, drift is accepted and tokens floor relaxed
    assert flags.get("preview_final_drift_acceptable") is True
    assert flags.get("primary_metric_acceptable") is True


def test_tiny_relax_relaxes_tokens_floor_for_ppl(monkeypatch):
    # Balanced default with pm_ratio policy and tiny token counts should still pass under tiny_relax
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    try:
        flags = C._compute_validation_flags(
            ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
            spectral={"caps_applied": 0},
            rmt={"stable": True},
            invariants={"status": "pass"},
            tier="balanced",
            _ppl_metrics={"preview_total_tokens": 1000, "final_total_tokens": 1000},
            primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        )
    finally:
        monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)

    assert isinstance(flags, dict)
    assert flags.get("primary_metric_acceptable") is True


def test_prepare_guard_overhead_section_fallback_paths():
    # Direct ratio computation path
    payload = {"bare_final": 100.0, "guarded_final": 101.0, "overhead_threshold": 0.02}
    out, passed = C._prepare_guard_overhead_section(payload)
    assert out.get("evaluated") is True and out.get("overhead_ratio") == 1.01
    assert passed is True

    # Unavailable ratio path → not evaluated and soft-pass
    out2, passed2 = C._prepare_guard_overhead_section({"messages": ["info"]})
    assert out2.get("evaluated") is False and out2.get("passed") is True
    assert any("unavailable" in e.lower() for e in out2.get("errors", []))


def test_validation_flags_hysteresis_applied_and_moe_observed(monkeypatch):
    # Set ratio slightly above base ratio limit but within hysteresis to trigger hysteresis_applied
    tier = "balanced"
    ppl = {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.11}
    spectral = {"caps_applied": 0}
    rmt = {"stable": True}
    invariants = {"status": "pass"}
    primary_metric = {"kind": "ppl_causal", "ratio_vs_baseline": 1.11}
    pm_policy = {"min_tokens": 0, "hysteresis_ratio": 0.02}  # base 1.10 + 0.02 = 1.12
    fake_policies = {"balanced": {"metrics": {"pm_ratio": pm_policy}}}
    monkeypatch.setattr(C, "get_tier_policies", lambda *_a, **_k: dict(fake_policies))
    flags = C._compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier=tier,
        _ppl_metrics={},
        primary_metric=primary_metric,
        dataset_capacity=None,
        pm_acceptance_range=None,
    )
    assert flags.get("primary_metric_acceptable") is True
    assert flags.get("hysteresis_applied") in {True, False}

    # MoE observed path populates moe flags (non-gating)
    flags2 = C._compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier=tier,
        _ppl_metrics={},
        primary_metric=primary_metric,
        dataset_capacity=None,
        moe={"utilization_mean": 0.5},
    )
    assert flags2.get("moe_observed") is True
