from __future__ import annotations

import pytest

from invarlock.core.runner_pairing import (
    BOOTSTRAP_COVERAGE_REQUIREMENTS,
    assess_bootstrap_coverage,
    compare_with_baseline,
    compute_window_pairing_metrics,
    duplicate_fraction,
    overlap_fraction_from_context,
)


def test_duplicate_fraction_paths() -> None:
    assert duplicate_fraction([]) == 0.0
    assert duplicate_fraction([[]]) == 0.0
    assert duplicate_fraction([[1, 2, 3], [1, 2, 3]]) == pytest.approx(0.5)
    assert duplicate_fraction([[1, 2, 3], [3, 2, 1]]) == 0.0
    assert (
        duplicate_fraction(
            [[1, 2, 3], [1, 2, 3]],
            labels=[[-100, 10, -100], [-100, 11, -100]],
        )
        == 0.0
    )
    assert duplicate_fraction(
        [[1, 2, 3], [1, 2, 3]],
        labels=[[-100, 10, -100], [-100, 10, -100]],
    ) == pytest.approx(0.5)


def test_overlap_fraction_from_context_paths() -> None:
    assert overlap_fraction_from_context(None) is None
    assert overlap_fraction_from_context({"dataset": []}) is None
    assert overlap_fraction_from_context({"dataset": {"stride": 1}}) is None
    assert (
        overlap_fraction_from_context({"dataset": {"seq_len": "bad", "stride": 1}})
        is None
    )
    assert (
        overlap_fraction_from_context({"dataset": {"seq_len": 0, "stride": 1}}) is None
    )
    assert (
        overlap_fraction_from_context({"dataset": {"seq_len": 4, "stride": -1}}) is None
    )
    assert overlap_fraction_from_context(
        {"dataset": {"seq_len": 4, "stride": 2}}
    ) == pytest.approx(0.5)
    assert (
        overlap_fraction_from_context({"dataset": {"seq_len": 4, "stride": 6}}) == 0.0
    )
    assert (
        overlap_fraction_from_context({"dataset": {"seq_len": 4, "stride": None}})
        is None
    )


def test_compare_with_baseline_paths() -> None:
    no_baseline = compare_with_baseline([0], [[1, 2, 3]], None, "preview")
    assert no_baseline["reason"] == "no_baseline_reference"

    invalid_baseline = compare_with_baseline(
        [0], [[1, 2, 3]], {"window_ids": "bad", "input_ids": []}, "preview"
    )
    assert invalid_baseline["reason"] == "invalid_baseline_reference"

    baseline_with_invalid_ids = compare_with_baseline(
        [0],
        [[1, 2, 3]],
        {"window_ids": [object(), 0], "input_ids": [[9, 9, 9], [1, 2, 3]]},
        "preview",
    )
    assert baseline_with_invalid_ids["matched"] == 0
    assert baseline_with_invalid_ids["expected"] == 2
    assert baseline_with_invalid_ids["reason"] == "invalid_baseline_reference"

    stats = compare_with_baseline(
        [0, 1, "x"],
        [[1, 2, 3], [9, 9, 9], [1, 1, 1]],
        {"window_ids": [0, 1, 2], "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
        "preview",
    )
    assert stats["matched"] == 1
    assert stats["mismatched_ids"] == [1]
    assert stats["missing_ids"] == [2]
    assert stats["unexpected_ids"] == ["x"]
    assert stats["reason"].startswith("preview_missing_ids")

    unexpected = compare_with_baseline(
        [5],
        [[1, 2, 3]],
        {"window_ids": [0], "input_ids": [[1, 2, 3]]},
        "final",
    )
    assert unexpected["unexpected_ids"] == [5]
    assert unexpected["reason"].startswith("final_missing_ids")

    unexpected_only = compare_with_baseline(
        [5, 6],
        [[1, 2, 3], [4, 5, 6]],
        {"window_ids": [5], "input_ids": [[1, 2, 3]]},
        "final",
    )
    assert unexpected_only["unexpected_ids"] == [6]
    assert unexpected_only["missing_ids"] == []
    assert unexpected_only["reason"].startswith("final_unexpected_ids")

    label_match = compare_with_baseline(
        [7],
        [[1, 2, 3]],
        {
            "window_ids": [7],
            "input_ids": [[1, 2, 3]],
            "labels": [[-100, 10, -100]],
        },
        "preview",
        run_labels=[[-100, 10, -100]],
    )
    assert label_match["matched"] == 1
    assert label_match["reason"] is None

    label_mismatch = compare_with_baseline(
        [7],
        [[1, 2, 3]],
        {
            "window_ids": [7],
            "input_ids": [[1, 2, 3]],
            "labels": [[-100, 10, -100]],
        },
        "preview",
        run_labels=[[-100, 11, -100]],
    )
    assert label_mismatch["matched"] == 0
    assert label_mismatch["mismatched_ids"] == [7]
    assert label_mismatch["reason"].startswith("preview_token_mismatch")


def test_compute_window_pairing_metrics_paths() -> None:
    no_baseline = compute_window_pairing_metrics(
        preview_window_ids=[0],
        preview_tokens=[[1, 2, 3]],
        final_window_ids=[1],
        final_tokens=[[4, 5, 6]],
        pairing_context=None,
        config_context=None,
        preview_batches=1,
        final_batches=1,
    )
    assert no_baseline["match_fraction"] == 1.0
    assert no_baseline["overlap_fraction"] == 1.0
    assert no_baseline["reason"] == "overlap_unknown"

    with_baseline = compute_window_pairing_metrics(
        preview_window_ids=[0, 1],
        preview_tokens=[[1, 2, 3], [1, 2, 3]],
        final_window_ids=[2],
        final_tokens=[[9, 9, 9]],
        pairing_context={
            "preview": {"window_ids": [0, 1], "input_ids": [[1, 2, 3], [8, 8, 8]]},
            "final": {"window_ids": [2], "input_ids": [[9, 9, 9]]},
        },
        config_context={"dataset": {"seq_len": 4, "stride": 2}},
        preview_batches=2,
        final_batches=1,
    )
    assert with_baseline["match_fraction"] < 1.0
    assert with_baseline["overlap_fraction"] == pytest.approx(0.5)
    assert with_baseline["duplicate_fraction"] > 0.0
    assert with_baseline["count_mismatch"] is True
    assert with_baseline["reason"] == "preview_token_mismatch:[1]"

    matched_non_overlapping = compute_window_pairing_metrics(
        preview_window_ids=[0],
        preview_tokens=[[1, 2, 3]],
        final_window_ids=[1],
        final_tokens=[[4, 5, 6]],
        pairing_context={
            "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
            "final": {"window_ids": [1], "input_ids": [[4, 5, 6]]},
        },
        config_context={"dataset": {"seq_len": 4, "stride": 4}},
        preview_batches=1,
        final_batches=1,
    )
    assert matched_non_overlapping["match_fraction"] == 1.0
    assert matched_non_overlapping["duplicate_fraction"] == 0.0
    assert matched_non_overlapping["reason"] is None

    overlapping_only = compute_window_pairing_metrics(
        preview_window_ids=[0],
        preview_tokens=[[1, 2, 3]],
        final_window_ids=[1],
        final_tokens=[[4, 5, 6]],
        pairing_context={
            "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
            "final": {"window_ids": [1], "input_ids": [[4, 5, 6]]},
        },
        config_context={"dataset": {"seq_len": 4, "stride": 2}},
        preview_batches=1,
        final_batches=1,
    )
    assert overlapping_only["reason"] == "overlapping_windows"

    duplicate_only = compute_window_pairing_metrics(
        preview_window_ids=[0],
        preview_tokens=[[1, 2, 3]],
        final_window_ids=[1],
        final_tokens=[[1, 2, 3]],
        pairing_context={
            "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
            "final": {"window_ids": [1], "input_ids": [[1, 2, 3]]},
        },
        config_context={"dataset": {"seq_len": 4, "stride": 4}},
        preview_batches=1,
        final_batches=1,
    )
    assert duplicate_only["reason"] == "duplicate_windows"

    count_mismatch_only = compute_window_pairing_metrics(
        preview_window_ids=[0],
        preview_tokens=[[1, 2, 3]],
        final_window_ids=[1],
        final_tokens=[[4, 5, 6]],
        pairing_context={
            "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
            "final": {"window_ids": [1], "input_ids": [[4, 5, 6]]},
        },
        config_context={"dataset": {"seq_len": 4, "stride": 4}},
        preview_batches=1,
        final_batches=2,
    )
    assert count_mismatch_only["reason"] == "count_mismatch"

    zero_expected = compute_window_pairing_metrics(
        preview_window_ids=[],
        preview_tokens=[],
        preview_labels=[],
        final_window_ids=[],
        final_tokens=[],
        final_labels=[],
        pairing_context={
            "preview": {"window_ids": [], "input_ids": []},
            "final": {"window_ids": [], "input_ids": []},
        },
        config_context={"dataset": {"seq_len": 4, "stride": 4}},
        preview_batches=0,
        final_batches=0,
    )
    assert zero_expected["match_fraction"] == 1.0
    assert zero_expected["reason"] is None

    mlm_distinct = compute_window_pairing_metrics(
        preview_window_ids=[0],
        preview_tokens=[[1, 2, 3]],
        preview_labels=[[-100, 10, -100]],
        final_window_ids=[1],
        final_tokens=[[1, 2, 3]],
        final_labels=[[-100, 11, -100]],
        pairing_context={
            "preview": {
                "window_ids": [0],
                "input_ids": [[1, 2, 3]],
                "labels": [[-100, 10, -100]],
            },
            "final": {
                "window_ids": [1],
                "input_ids": [[1, 2, 3]],
                "labels": [[-100, 11, -100]],
            },
        },
        config_context={"dataset": {"seq_len": 4, "stride": 4}},
        preview_batches=1,
        final_batches=1,
    )
    assert mlm_distinct["match_fraction"] == 1.0
    assert mlm_distinct["duplicate_fraction"] == 0.0
    assert mlm_distinct["reason"] is None


def test_assess_bootstrap_coverage_paths() -> None:
    summary = assess_bootstrap_coverage(
        tier="balanced",
        preview_batches=200,
        final_batches=200,
        bootstrap_enabled=True,
        bootstrap_replicates=1500,
    )
    assert summary["ok"] is True
    assert (
        summary["coverage"]["preview"]["required"]
        == BOOTSTRAP_COVERAGE_REQUIREMENTS["balanced"]["preview"]
    )

    not_ok = assess_bootstrap_coverage(
        tier="mystery",
        preview_batches=1,
        final_batches=1,
        bootstrap_enabled=False,
        bootstrap_replicates=1,
        requirements={"balanced": {"preview": 5, "final": 6, "replicates": 7}},
    )
    assert not_ok["ok"] is False
    assert not_ok["coverage"]["preview"]["ok"] is False
    assert not_ok["coverage"]["final"]["ok"] is False
    assert not_ok["coverage"]["replicates"]["ok"] is True

    zero_required = assess_bootstrap_coverage(
        tier="empty",
        preview_batches=0,
        final_batches=0,
        bootstrap_enabled=True,
        bootstrap_replicates=0,
        requirements={
            "balanced": {"preview": 5, "final": 6, "replicates": 7},
            "empty": {},
        },
    )
    assert zero_required["ok"] is True
    assert zero_required["coverage"]["preview"]["ok"] is True
    assert zero_required["coverage"]["replicates"]["ok"] is True
