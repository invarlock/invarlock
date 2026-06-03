from __future__ import annotations

from invarlock.reporting.report_validation import compute_validation_flags


def test_compute_validation_flags_rejects_bool_ratio_for_ppl_metric() -> None:
    flags = compute_validation_flags(
        ppl={"ratio_vs_baseline": True, "preview_final_ratio": 1.0},
        spectral={"caps_applied": 0, "max_caps": 5},
        rmt={"stable": True},
        invariants={"status": "pass"},
        primary_metric={"kind": "ppl_causal", "ratio_vs_baseline": True},
        get_tier_policies_fn=lambda: {"balanced": {"metrics": {"pm_ratio": {}}}},
    )
    assert flags["primary_metric_acceptable"] is False


def test_compute_validation_flags_rejects_bool_sample_size_for_accuracy() -> None:
    flags = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "preview_final_ratio": 1.0},
        spectral={"caps_applied": 0, "max_caps": 5},
        rmt={"stable": True},
        invariants={"status": "pass"},
        primary_metric={"kind": "accuracy", "ratio_vs_baseline": 0.0, "n_final": True},
        get_tier_policies_fn=lambda: {
            "balanced": {
                "metrics": {"accuracy": {"delta_min_pp": -1.0, "min_examples": 1}}
            }
        },
    )
    assert flags["primary_metric_acceptable"] is False
