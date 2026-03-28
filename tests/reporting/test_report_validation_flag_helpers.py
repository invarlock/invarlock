from __future__ import annotations

from invarlock.reporting import report_validation as validation_mod


def test_compute_validation_flags_accuracy_hysteresis(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.05,
                    "min_tokens": 100,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.3,
                    "min_examples": 20,
                    "min_examples_fraction": 0.4,
                    "hysteresis_delta_pp": 0.15,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)

    ppl = {
        "preview_final_ratio": 1.0,
        "ratio_vs_baseline": 1.12,
        "ratio_ci": (1.05, 1.13),
    }
    spectral = {"caps_applied": 4, "caps_exceeded": True, "summary": {}}
    rmt = {"stable": True}
    invariants = {"status": "ok"}
    guard_overhead = {"overhead_ratio": 1.12, "overhead_threshold": 0.05}
    primary_metric = {"kind": "accuracy", "ratio_vs_baseline": 0.2, "n_final": 10}
    dataset_capacity = {"examples_available": 40}
    ppl_metrics = {"preview_total_tokens": 60, "final_total_tokens": 60}

    flags = validation_mod.compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier="balanced",
        _ppl_metrics=ppl_metrics,
        guard_overhead=guard_overhead,
        primary_metric=primary_metric,
        dataset_capacity=dataset_capacity,
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert flags["preview_final_drift_acceptable"] is True
    assert flags["invariants_pass"] is True
    assert flags["rmt_stable"] is True
    assert flags["spectral_stable"] is False
    assert flags["guard_overhead_acceptable"] is False
    assert flags["primary_metric_acceptable"] is False
    assert flags["hysteresis_applied"] is True
    assert flags["primary_metric_tail_acceptable"] is True


def test_compute_validation_flags_core_gates_ppl_and_tail_fail(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                    "ratio_limit_base": 1.10,
                },
            },
            "spectral": {"max_caps": 1},
        }
    }
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)

    flags = validation_mod.compute_validation_flags(
        {
            "preview_final_ratio": 1.20,
            "ratio_vs_baseline": 1.25,
            "ratio_ci": (1.20, 1.30),
        },
        {"caps_applied": 2, "max_caps": 1, "caps_exceeded": False, "summary": {}},
        {"stable": False},
        {"status": "fail"},
        tier="balanced",
        guard_overhead={"overhead_ratio": 1.20, "overhead_threshold": 0.0},
        pm_tail={"mode": "fail", "evaluated": True, "passed": False},
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert flags["preview_final_drift_acceptable"] is False
    assert flags["primary_metric_acceptable"] is False
    assert flags["invariants_pass"] is False
    assert flags["spectral_stable"] is False
    assert flags["rmt_stable"] is False
    assert flags["guard_overhead_acceptable"] is False
    assert flags["primary_metric_tail_acceptable"] is False


def test_compute_validation_flags_tiny_relax_allows_unevaluated_overhead(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.0,
                    "min_examples": 0,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 5},
        }
    }
    guard_overhead = {"passed": False, "evaluated": False, "errors": ["missing"]}
    flags = validation_mod.compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead=guard_overhead,
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        _ppl_metrics={"preview_total_tokens": 0, "final_total_tokens": 0},
        dataset_capacity={"tokens_available": 0},
        tiny_relax=True,
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert flags["guard_overhead_acceptable"] is True


def test_compute_validation_flags_guard_overhead_ratio_failure(monkeypatch):
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.0,
                    "min_examples": 0,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    guard_overhead = {"overhead_ratio": 1.05, "overhead_threshold": 0.01}
    flags = validation_mod.compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead=guard_overhead,
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"tokens_available": 20},
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert flags["guard_overhead_acceptable"] is False


def test_compute_validation_flags_guard_overhead_ratio_passes_with_tiny_relax(
    monkeypatch,
):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.0,
                    "min_examples": 0,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    guard_overhead = {"overhead_ratio": 1.05, "overhead_threshold": 0.01}
    flags = validation_mod.compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.2},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead=guard_overhead,
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.2},
        _ppl_metrics={"preview_total_tokens": 0, "final_total_tokens": 0},
        dataset_capacity={"tokens_available": 0},
        tiny_relax=True,
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert flags["guard_overhead_acceptable"] is True


def test_compute_validation_flags_guard_overhead_passes_when_ratio_missing(monkeypatch):
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.0,
                    "min_examples": 0,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    guard_overhead = {"overhead_threshold": 0.01}
    flags = validation_mod.compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 0.9},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead=guard_overhead,
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 0.9},
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"tokens_available": 20},
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert flags["guard_overhead_acceptable"] is True


def test_compute_validation_flags_accuracy_fails_low_examples(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": -0.2,
                    "min_examples": 200,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)

    primary_metric = {"kind": "accuracy", "ratio_vs_baseline": -0.1, "n_final": 50}
    flags = validation_mod.compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead={"passed": True},
        primary_metric=primary_metric,
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"tokens_available": 20},
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert flags["primary_metric_acceptable"] is False


def test_compute_validation_flags_accuracy_respects_dataset_fraction_floor(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": -0.5,
                    "min_examples": 0,
                    "min_examples_fraction": 0.25,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)

    primary_metric = {"kind": "accuracy", "ratio_vs_baseline": -0.3, "n_final": 15}
    dataset_capacity = {"examples_available": 80}
    flags = validation_mod.compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead={"passed": True},
        primary_metric=primary_metric,
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity=dataset_capacity,
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert flags["primary_metric_acceptable"] is False


def test_compute_validation_flags_accuracy_passes_with_hysteresis(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": -1.0,
                    "min_examples": 50,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.5,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    monkeypatch.delenv("INVARLOCK_TINY_RELAX", raising=False)

    primary_metric = {"kind": "accuracy", "ratio_vs_baseline": -1.2, "n_final": 80}
    flags = validation_mod.compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead={"passed": True},
        primary_metric=primary_metric,
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"examples_available": 200},
        get_tier_policies_fn=lambda: dict(fake_policies),
    )

    assert flags["primary_metric_acceptable"] is True
    assert flags["hysteresis_applied"] is True


def test_compute_validation_flags_marks_moe_observed():
    flags = validation_mod.compute_validation_flags(
        {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0},
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        moe={"top_k": 1},
        guard_overhead={"passed": True},
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": 1.0},
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"tokens_available": 20},
    )
    assert flags["moe_observed"] is True
    assert flags["moe_identity_ok"] is True


def test_compute_validation_flags_reconciles_ppl_primary_metric_ratio(monkeypatch):
    fake_policies = {
        "balanced": {
            "metrics": {
                "pm_ratio": {
                    "hysteresis_ratio": 0.0,
                    "min_tokens": 0,
                    "min_token_fraction": 0.0,
                },
                "accuracy": {
                    "delta_min_pp": 0.0,
                    "min_examples": 0,
                    "min_examples_fraction": 0.0,
                    "hysteresis_delta_pp": 0.0,
                },
            },
            "spectral": {"max_caps": 3},
        }
    }
    ppl = {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.2}
    primary_metric = {"kind": "ppl_causal", "ratio_vs_baseline": 1.05}
    flags = validation_mod.compute_validation_flags(
        ppl,
        {"caps_applied": 0},
        {"stable": True},
        {"status": "ok"},
        tier="balanced",
        guard_overhead={"passed": True},
        primary_metric=primary_metric,
        _ppl_metrics={"preview_total_tokens": 10, "final_total_tokens": 10},
        dataset_capacity={"tokens_available": 20},
        get_tier_policies_fn=lambda: dict(fake_policies),
    )
    assert flags["primary_metric_acceptable"] is True
