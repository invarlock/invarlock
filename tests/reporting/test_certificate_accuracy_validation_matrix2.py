from __future__ import annotations

from invarlock.reporting.certificate import _compute_validation_flags


def test_accuracy_hysteresis_applied_and_accepts_conservative() -> None:
    # Conservative tier: delta_min_pp = -0.5, hysteresis_delta_pp = 0.1
    # Set delta = -0.55 → meets_delta via hysteresis; expect hysteresis_applied True
    flags = _compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={},
        rmt={},
        invariants={},
        tier="conservative",
        _ppl_metrics={"preview_total_tokens": 60000, "final_total_tokens": 60000},
        target_ratio=None,
        guard_overhead={},
        primary_metric={
            "kind": "accuracy",
            # ratio_vs_baseline used as Δ proportion for accuracy kinds
            "ratio_vs_baseline": -0.55 / 100.0 * 100.0
            if False
            else -0.55,  # -0.55 pp interpreted as -0.55
            "n_final": 1000,
        },
        moe={},
        dataset_capacity={"examples_available": 100000},
    )
    assert flags["primary_metric_acceptable"] is True
    assert flags.get("hysteresis_applied") is True


def test_accuracy_min_examples_fraction_precedence_balanced() -> None:
    # Balanced: min_examples=200, min_examples_fraction=0.01 → eff_min = max(200, 1% of examples_available)
    # With examples_available=50_000 → eff_min=500; n_final below fails; above passes
    common_kwargs = {
        "ppl": {"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        "spectral": {},
        "rmt": {},
        "invariants": {},
        "tier": "balanced",
        "_ppl_metrics": {"preview_total_tokens": 60000, "final_total_tokens": 60000},
        "target_ratio": None,
        "guard_overhead": {},
        "moe": {},
        "dataset_capacity": {"examples_available": 50_000},
    }

    # Below floor
    flags_fail = _compute_validation_flags(
        **common_kwargs,
        primary_metric={"kind": "accuracy", "ratio_vs_baseline": 0.0, "n_final": 400},
    )
    assert flags_fail["primary_metric_acceptable"] is False

    # Above floor
    flags_pass = _compute_validation_flags(
        **common_kwargs,
        primary_metric={"kind": "accuracy", "ratio_vs_baseline": 0.0, "n_final": 600},
    )
    assert flags_pass["primary_metric_acceptable"] is True
