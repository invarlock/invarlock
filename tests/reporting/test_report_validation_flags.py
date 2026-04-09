from __future__ import annotations

from invarlock.reporting import report_overhead as report_overhead_mod
from invarlock.reporting import report_policy as report_policy_mod
from invarlock.reporting import report_validation as report_validation_mod


def test_compute_validation_flags_tiny_relax_and_tokens_floor(monkeypatch):
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
    flags = report_validation_mod.compute_validation_flags(
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
        tiny_relax=True,
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert isinstance(flags, dict)
    # With tiny relax, drift is accepted and tokens floor relaxed
    assert flags.get("preview_final_drift_acceptable") is True
    assert flags.get("primary_metric_acceptable") is True


def test_compute_validation_flags_ignores_env_tiny_relax_without_provenance(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "yes")

    flags = report_validation_mod.compute_validation_flags(
        ppl={"preview_final_ratio": 2.0, "ratio_vs_baseline": 1.0},
        spectral={"caps_applied": 0},
        rmt={"stable": True},
        invariants={"status": "pass"},
        tier="balanced",
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
    )

    assert flags.get("preview_final_drift_acceptable") is False


def test_compute_validation_flags_rejects_bool_preview_final_ratio() -> None:
    flags = report_validation_mod.compute_validation_flags(
        ppl={"preview_final_ratio": True, "ratio_vs_baseline": 1.0},
        spectral={"caps_applied": 0},
        rmt={"stable": True},
        invariants={"status": "pass"},
        tier="balanced",
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
    )

    assert flags["preview_final_drift_acceptable"] is False


def test_tiny_relax_relaxes_tokens_floor_for_ppl():
    # Balanced default with pm_ratio policy and tiny token counts should still pass under tiny_relax
    flags = report_validation_mod.compute_validation_flags(
        ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        spectral={"caps_applied": 0},
        rmt={"stable": True},
        invariants={"status": "pass"},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 1000, "final_total_tokens": 1000},
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        tiny_relax=True,
    )

    assert isinstance(flags, dict)
    assert flags.get("primary_metric_acceptable") is True


def test_compute_validation_flags_does_not_treat_target_ratio_as_gate_cap() -> None:
    flags = report_validation_mod.compute_validation_flags(
        ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.03},
        spectral={"caps_applied": 0},
        rmt={"stable": True},
        invariants={"status": "pass"},
        tier="balanced",
        target_ratio=1.0,
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.03},
    )

    assert isinstance(flags, dict)
    assert flags.get("primary_metric_acceptable") is True


def test_compute_validation_flags_handles_policy_cast_failures(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)
    fake_policies = {
        "balanced": {"metrics": {"pm_ratio": {"ratio_limit_base": "bad"}}},
    }
    flags = report_validation_mod.compute_validation_flags(
        ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        spectral={"caps_applied": 0},
        rmt={"stable": True},
        invariants={"status": "pass"},
        tier="balanced",
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        pm_acceptance_range={"min": "bad", "max": "bad"},
        pm_drift_band={"min": 2.0, "max": 1.0},
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert isinstance(flags, dict)
    assert flags.get("preview_final_drift_acceptable") is True
    assert flags.get("primary_metric_acceptable") is True


def test_compute_validation_flags_applies_min_tokens_tolerance(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)
    pm_policy = {
        "ratio_limit_base": 1.10,
        "min_tokens": 100,
        "min_tokens_tolerance": 0.10,
        "min_token_fraction": 0.0,
    }
    fake_policies = {
        "balanced": {"metrics": {"pm_ratio": pm_policy}},
    }
    flags = report_validation_mod.compute_validation_flags(
        ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        spectral={"caps_applied": 0},
        rmt={"stable": True},
        invariants={"status": "pass"},
        tier="balanced",
        _ppl_metrics={
            "preview_total_tokens": 47,
            "final_total_tokens": 48,
            "bootstrap": {
                "coverage": {
                    "preview": {"used": 50, "required": 50},
                    "final": {"used": 50, "required": 50},
                }
            },
        },
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert isinstance(flags, dict)
    assert flags.get("primary_metric_acceptable") is True


def test_compute_validation_flags_handles_bad_min_tokens_tolerance(monkeypatch) -> None:
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)
    pm_policy = {
        "ratio_limit_base": 1.10,
        "min_tokens": 100,
        "min_tokens_tolerance": "bad",
        "min_token_fraction": 0.0,
    }
    fake_policies = {
        "balanced": {"metrics": {"pm_ratio": pm_policy}},
    }
    flags = report_validation_mod.compute_validation_flags(
        ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        spectral={"caps_applied": 0},
        rmt={"stable": True},
        invariants={"status": "pass"},
        tier="balanced",
        _ppl_metrics={
            "preview_total_tokens": 47,
            "final_total_tokens": 48,
            "bootstrap": {
                "coverage": {
                    "preview": {"used": 50, "required": 50},
                    "final": {"used": 50, "required": 50},
                }
            },
        },
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert isinstance(flags, dict)
    assert flags.get("primary_metric_acceptable") is False


def test_compute_validation_flags_clamps_negative_min_tokens_tolerance(
    monkeypatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)
    pm_policy = {
        "ratio_limit_base": 1.10,
        "min_tokens": 100,
        "min_tokens_tolerance": -0.5,
        "min_token_fraction": 0.0,
    }
    fake_policies = {
        "balanced": {"metrics": {"pm_ratio": pm_policy}},
    }
    flags = report_validation_mod.compute_validation_flags(
        ppl={"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        spectral={"caps_applied": 0},
        rmt={"stable": True},
        invariants={"status": "pass"},
        tier="balanced",
        _ppl_metrics={
            "preview_total_tokens": 47,
            "final_total_tokens": 48,
            "bootstrap": {
                "coverage": {
                    "preview": {"used": 50, "required": 50},
                    "final": {"used": 50, "required": 50},
                }
            },
        },
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert isinstance(flags, dict)
    assert flags.get("primary_metric_acceptable") is False


def test_resolve_tiny_relax_from_report_context_and_auto_fields() -> None:
    assert (
        report_policy_mod.resolve_tiny_relax_from_report(
            {"context": {"run": {"tiny_relax": "on"}}}
        )
        is True
    )
    assert (
        report_policy_mod.resolve_tiny_relax_from_report(
            {"context": {"eval": {"tiny_relax": 1}}}
        )
        is True
    )
    assert (
        report_policy_mod.resolve_tiny_relax_from_report({"auto": {"tiny_relax": True}})
        is True
    )
    assert (
        report_policy_mod.resolve_tiny_relax_from_report(
            {"meta": {"auto": {"tiny_relax": "maybe"}}}
        )
        is False
    )


def test_prepare_guard_overhead_section_fallback_paths():
    # Direct ratio computation path
    payload = {"bare_ppl": 100.0, "guarded_ppl": 101.0, "overhead_threshold": 0.02}
    out, passed = report_overhead_mod.prepare_guard_overhead_section(payload)
    assert out.get("evaluated") is True and out.get("overhead_ratio") == 1.01
    assert passed is True

    # Unavailable ratio path → not evaluated and soft-pass
    out2, passed2 = report_overhead_mod.prepare_guard_overhead_section(
        {"source": "unit"}
    )
    assert out2.get("evaluated") is False and out2.get("passed") is True
    assert any(
        "unavailable" in item.get("message", "").lower()
        for item in out2.get("diagnostics", [])
    )
    assert "errors" not in out2


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
    flags = report_validation_mod.compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier=tier,
        _ppl_metrics={},
        primary_metric=primary_metric,
        dataset_capacity=None,
        pm_acceptance_range=None,
        get_tier_policies_fn=lambda: dict(fake_policies),
    )
    assert flags.get("primary_metric_acceptable") is True
    assert flags.get("hysteresis_applied") in {True, False}

    # MoE observed path populates moe flags (non-gating)
    flags2 = report_validation_mod.compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier=tier,
        _ppl_metrics={},
        primary_metric=primary_metric,
        dataset_capacity=None,
        moe={"utilization_mean": 0.5},
        get_tier_policies_fn=lambda: dict(fake_policies),
    )
    assert flags2.get("moe_observed") is True
