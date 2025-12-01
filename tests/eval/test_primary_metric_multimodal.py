from __future__ import annotations

import math

from invarlock.eval.primary_metric import compute_primary_metric_from_report


def test_ppl_seq2seq_behaves_like_expected():
    # Windows correspond to ppl=exp(mean logloss) over token_counts
    report = {
        "evaluation_windows": {
            "preview": {"logloss": [math.log(5.0)], "token_counts": [10]},
            "final": {"logloss": [math.log(6.0)], "token_counts": [10]},
        }
    }
    baseline = {
        "metrics": {
            "primary_metric": {"kind": "ppl_seq2seq", "final": 5.0, "preview": 5.0}
        }
    }

    pm = compute_primary_metric_from_report(
        report, kind="ppl_seq2seq", baseline=baseline
    )

    assert math.isclose(pm["preview"], 5.0, rel_tol=1e-12)
    assert math.isclose(pm["final"], 6.0, rel_tol=1e-12)
    assert math.isclose(pm["ratio_vs_baseline"], 6.0 / 5.0, rel_tol=1e-12)


def test_vqa_accuracy_alias_behaves_like_accuracy():
    report = {
        "metrics": {
            "classification": {
                "preview": {"correct_total": 80, "total": 100},
                "final": {"correct_total": 190, "total": 200},
            }
        }
    }
    baseline = {
        "metrics": {
            "primary_metric": {"kind": "accuracy", "final": 0.90, "preview": 0.90}
        }
    }

    pm_alias = compute_primary_metric_from_report(
        report, kind="vqa_accuracy", baseline=baseline
    )
    pm_acc = compute_primary_metric_from_report(
        report, kind="accuracy", baseline=baseline
    )

    assert math.isclose(pm_alias["preview"], 0.80, rel_tol=1e-12)
    assert math.isclose(pm_alias["final"], 0.95, rel_tol=1e-12)
    # For accuracy we report delta vs baseline in ratio field
    assert math.isclose(pm_alias["ratio_vs_baseline"], 0.95 - 0.90, rel_tol=1e-12)
    # Matches accuracy baseline
    assert pm_alias["ratio_vs_baseline"] == pm_acc["ratio_vs_baseline"]
