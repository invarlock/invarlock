from __future__ import annotations

from invarlock.reporting.validation.report import compute_validation_flags


def test_guard_metric_impact_not_evaluated_with_errors_fails_closed() -> None:
    flags = compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "ratio_ci": (1.0, 1.0)},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 1000, "final_total_tokens": 1000},
        target_ratio=None,
        guard_metric_impact={
            "evaluated": False,
            "errors": ["missing"],
            "degradation": float("nan"),
        },
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 2000},
    )
    assert flags["guard_metric_impact_acceptable"] is False
