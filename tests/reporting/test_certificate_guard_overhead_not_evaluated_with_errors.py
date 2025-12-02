from __future__ import annotations

from invarlock.reporting.certificate import _compute_validation_flags


def test_guard_overhead_not_evaluated_with_errors_soft_pass() -> None:
    # When guard_overhead not evaluated or has errors, the flag soft-passes
    flags = _compute_validation_flags(
        ppl={"ratio_vs_baseline": 1.0, "ratio_ci": (1.0, 1.0)},
        spectral={},
        rmt={},
        invariants={},
        tier="balanced",
        _ppl_metrics={"preview_total_tokens": 1000, "final_total_tokens": 1000},
        target_ratio=None,
        # Non-finite ratio treated as pass in best-effort Compare & Certify flows
        guard_overhead={
            "evaluated": False,
            "errors": ["missing"],
            "overhead_ratio": float("nan"),
        },
        primary_metric=None,
        moe={},
        dataset_capacity={"tokens_available": 2000},
    )
    assert flags["guard_overhead_acceptable"] is True
