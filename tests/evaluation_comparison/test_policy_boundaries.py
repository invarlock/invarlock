from __future__ import annotations

import copy

import pytest

from invarlock.evaluation_comparison.comparison import compare_runs
from invarlock.evaluation_records.templates import example_project
from invarlock.evidence_pack_contract import (
    PAIRED_RECORDS_FORMAT,
    build_comparison_report,
)
from tests._policy_boundary_support import check_boundaries
from tests.evaluation_comparison.test_likelihood import policy as likelihood_policy
from tests.evaluation_comparison.test_likelihood import row, run


def test_native_acceptance_boundaries_and_pairwise_policy_conjunction():
    pairs = {
        "format": PAIRED_RECORDS_FORMAT,
        "metric": "exact_match",
        "schedule_sha256": "a" * 64,
        "records": [
            {"baseline": {"score": b}, "subject": {"score": s}}
            for b, s in zip(
                [1, 1, 0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 1, 1, 0, 0], strict=True
            )
        ],
    }
    defaults = {
        "delta_min_pp": -100.0,
        "minimum_record_count": 1,
        "maximum_interval_width_pp": 200.0,
        "minimum_side_accuracy": 0.0,
    }

    def compare(changes):
        return build_comparison_report(
            comparison_id="boundary-test",
            paired_records=pairs,
            policy={
                "resolved_policy": {"metrics": {"exact_match": defaults | changes}}
            },
            policy_digest="sha256:" + "b" * 64,
        )

    result = compare({})
    interval = result["uncertainty"]
    check_boundaries(
        {
            "delta_min_pp": interval["lower"],
            "minimum_record_count": 8,
            "maximum_interval_width_pp": interval["upper"] - interval["lower"],
            "minimum_side_accuracy": 0.625,
        },
        lambda changes: compare(changes)["verdict"] == "pass",
        relaxing_down={"delta_min_pp", "minimum_record_count", "minimum_side_accuracy"},
    )


@pytest.mark.parametrize("direction", ["higher", "lower"])
def test_captured_scalar_boundaries_and_pairwise_policy_conjunction(direction):
    baseline, subject, policy = example_project("judge")
    policy["slices"] = []
    policy["metrics"] = policy["metrics"][:1]
    metric = policy["metrics"][0]
    metric.update(
        direction=direction,
        minimum_count=1,
        maximum_regression=10.0,
        maximum_interval_width=10.0,
    )
    metric.pop("subject_minimum")
    for data, values in (
        (baseline, [1.0] * 6),
        (subject, [0.2, 0.5, 1.0, 1.25, 1.5, 2.0]),
    ):
        data["records"] = data["records"][:6]
        for record, value in zip(data["records"], values, strict=True):
            record["scores"]["quality"] = value

    def compare(changes):
        selected = copy.deepcopy(policy)
        selected["metrics"][0].update(changes)
        return compare_runs(baseline, subject, selected)["metrics"][0]

    result = compare({})
    interval = result["interval"]
    check_boundaries(
        {
            "minimum_count": 6,
            "maximum_regression": -interval["lower"]
            if direction == "higher"
            else interval["upper"],
            "maximum_interval_width": interval["upper"] - interval["lower"],
            "subject_minimum": result["subject_mean"],
            "subject_maximum": result["subject_mean"],
        },
        lambda changes: compare(changes)["decision"] == "pass",
        relaxing_down={"minimum_count", "subject_minimum"},
    )


def test_likelihood_ratio_boundaries_and_pairwise_policy_conjunction():
    baseline = run([row(str(i), -4.0 - i) for i in range(6)])
    subject = run([row(str(i), -4.5 - i * 0.8) for i in range(6)])
    policy = likelihood_policy()
    policy["metrics"][0].update(ratio_max=10.0, maximum_interval_width=10.0)

    def compare(changes):
        selected = copy.deepcopy(policy)
        selected["metrics"][0].update(changes)
        return compare_runs(baseline, subject, selected)["metrics"][0]

    result = compare({})
    interval = result["interval"]
    check_boundaries(
        {
            "minimum_count": 6,
            "ratio_max": interval["upper"],
            "maximum_interval_width": interval["upper"] - interval["lower"],
            "subject_minimum": result["subject_mean"],
            "subject_maximum": result["subject_mean"],
        },
        lambda changes: compare(changes)["decision"] == "pass",
        relaxing_down={"minimum_count", "subject_minimum"},
    )
