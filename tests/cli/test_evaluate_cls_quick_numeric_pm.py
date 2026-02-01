import math

from invarlock.eval.primary_metric import compute_primary_metric_from_report


def test_accuracy_pm_carries_counts_source_and_estimated_from_classification_pseudo():
    # Simulate a run report with classification aggregates but no windows/labels
    # and indicate pseudo counts were used.
    report = {
        "metrics": {
            "classification": {
                "preview": {"correct_total": 0, "total": 0},
                "final": {"correct_total": 10, "total": 10},
                "counts_source": "pseudo_config",
            }
        }
    }

    pm = compute_primary_metric_from_report(report, kind="accuracy", baseline=None)
    assert pm["kind"] == "accuracy"
    assert isinstance(pm.get("final"), int | float) and math.isfinite(
        float(pm["final"])
    )
    # New fields populated based on classification counts source
    assert pm.get("counts_source") == "pseudo_config"
    assert pm.get("estimated") is True
