from __future__ import annotations

from invarlock.reporting.run_pairing_contract import (
    build_dataset_window_stats,
    validate_pairing_report_metrics,
)


def test_validate_pairing_report_metrics_detects_pairing_violations() -> None:
    violations = validate_pairing_report_metrics(
        {
            "window_match_fraction": 0.75,
            "window_overlap_fraction": 0.5,
            "window_pairing_reason": "capacity_shortfall",
            "paired_windows": 0,
        },
        baseline_requested=True,
        profile="ci",
        preview_count_report=5,
        final_count_report=4,
        expected_preview=4,
        expected_final=4,
    )

    assert [violation.code for violation in violations] == [
        "E001",
        "E001",
        "E001",
        "E001",
        "E001",
    ]
    assert "window_match_fraction" in violations[0].message
    assert "window_overlap_fraction" in violations[1].message
    assert "window_pairing_reason" in violations[2].message
    assert "PAIRED-WINDOWS-COLLAPSED" in violations[3].message
    assert "counts do not match" in violations[4].message


def test_build_dataset_window_stats_maps_pairing_and_capacity_fields() -> None:
    stats = build_dataset_window_stats(
        match_fraction=1.0,
        overlap_fraction=0.0,
        window_plan={
            "coverage_ok": True,
            "preview_total_tokens": 120,
            "final_total_tokens": 140,
            "min_tokens_target": 200,
            "tokens_floor_met": True,
        },
    )

    assert stats == {
        "window_match_fraction": 1.0,
        "window_overlap_fraction": 0.0,
        "coverage": True,
        "preview_total_tokens": 120,
        "final_total_tokens": 140,
        "min_tokens_target": 200,
        "tokens_floor_met": True,
    }
