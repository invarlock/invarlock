from __future__ import annotations

from invarlock.reporting.report_enrichment import attach_classification
from invarlock.reporting.report_primary_metric_analysis import (
    build_primary_metric_analysis,
)


def test_attach_classification_carries_metrics_aggregates_for_verify() -> None:
    evaluation_report: dict[str, object] = {}
    report = {
        "metrics": {
            "classification": {
                "preview": {"correct_total": 1, "total": 1},
                "final": {"correct_total": 0, "total": 1},
                "counts_source": "measured",
            }
        }
    }

    attach_classification(evaluation_report, report)

    metrics = evaluation_report["metrics"]["classification"]
    assert metrics["counts_source"] == "measured"
    assert metrics["n_correct"] == 0
    assert metrics["n_total"] == 1
    assert metrics["estimated"] is False


def test_build_primary_metric_analysis_populates_multimodal_coverage_and_pairing() -> (
    None
):
    report = {
        "metrics": {
            "primary_metric": {
                "kind": "vqa_accuracy",
                "preview": 1.0,
                "final": 1.0,
                "ratio_vs_baseline": 0.0,
                "n_preview": 1,
                "n_final": 1,
            },
            "classification": {
                "preview": {"correct_total": 1, "total": 1},
                "final": {"correct_total": 1, "total": 1},
                "counts_source": "measured",
            },
        },
        "data": {"preview_n": 1, "final_n": 1},
        "evaluation_windows": {
            "preview": {
                "example_ids": ["red-square"],
                "records": [{"id": "red-square", "correct": True}],
            },
            "final": {
                "example_ids": ["green-square"],
                "records": [{"id": "green-square", "correct": True}],
            },
        },
        "meta": {"auto": {"tier": "balanced"}},
    }
    baseline_normalized = {
        "evaluation_windows": {
            "final": {
                "example_ids": ["green-square"],
                "records": [{"id": "green-square", "correct": True}],
            }
        }
    }
    baseline_ref = {"primary_metric": {"final": 1.0}}
    dataset_info = {"windows": {"preview": 1, "final": 1}}

    analysis, _ = build_primary_metric_analysis(
        report,
        baseline_normalized,
        baseline_ref,
        dataset_info,
    )

    stats = analysis["stats"]
    assert stats["paired_windows"] == 1
    assert stats["coverage"]["preview"]["used"] == 1
    assert stats["coverage"]["final"]["used"] == 1
    assert stats["window_match_fraction"] == 1.0
    assert stats["window_overlap_fraction"] == 0.0
