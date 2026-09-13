"""Captured summaries report retained numbers without inventing counts or policy."""

from copy import deepcopy

import pytest

from invarlock import captured_reporting, record_reporting
from invarlock.evaluation_records.templates import example_project
from tests._evaluation_support import build_pack, pack_json


def binary_comparison():
    baseline, subject, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    snapshot = build_pack(baseline, subject, policy)
    return snapshot, pack_json(snapshot, "report"), policy


def test_single_metric_summary_includes_values_interval_and_bound_requirements():
    snapshot, comparison, _ = binary_comparison()
    view = record_reporting._view(comparison, snapshot)
    metric = view.metrics[0]
    assert (
        f"baseline {metric.baseline}; subject {metric.candidate}; change {metric.change}"
        in view.summary
    )
    assert "95% interval:" in view.summary
    assert "All configured checks passed" not in view.summary
    assert "Matches:" not in view.summary
    assert "The policy requires the interval to stay at or above" in view.summary
    assert "The subject score must be at least 80%" in view.summary
    assert "No absolute minimum score" not in view.summary
    assert metric.interval.neutral == 0
    assert metric.interval.threshold_direction == "minimum"


def test_absent_floor_claim_requires_supplied_bound_policy():
    baseline, subject, policy = example_project("classification")
    policy["metrics"] = policy["metrics"][:1]
    policy["slices"] = []
    policy["metrics"][0].pop("subject_minimum")
    snapshot = build_pack(baseline, subject, policy)
    comparison = pack_json(snapshot, "report")
    bound = record_reporting._view(comparison, snapshot)
    unbound = record_reporting._view(comparison, None)
    assert "No absolute minimum score is required by this policy." in bound.summary
    assert "No absolute minimum score" not in str(unbound)
    assert unbound.metrics[0].interval.threshold_direction is None


def test_summary_aggregates_scopes_without_adding_overlapping_pair_counts():
    snapshot, comparison, _ = binary_comparison()
    comparison["metrics"] = [
        {**comparison["metrics"][0], "slice": scope, "decision": decision}
        for scope, decision in (
            ("overall", "pass"),
            ("west", "regression"),
            ("east", "insufficient_evidence"),
        )
    ]
    metrics = record_reporting._metric_views(comparison, {})
    summary = record_reporting._captured_summary(metrics)
    assert summary.startswith(
        "3 metric / scope results: 1 passed, 1 did not meet policy, and 1 need more evidence."
    )
    assert "overlapping slice counts must not be added together" in summary


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing",
        "nonbinary",
        "aggregation",
        "rounded_mean",
        "out_of_range",
        "unavailable",
        "huge_count",
        "inexact_count",
    ],
)
def test_match_counts_require_complete_binary_mean_with_exact_integer_arithmetic(
    mutation,
):
    _, comparison, _ = binary_comparison()
    metric = comparison["metrics"][0]
    metric.update(
        kind="exact_match", count=400, baseline_mean=0.05, subject_mean=0.06, delta=0.01
    )
    if mutation == "missing":
        metric["missing_ids"] = ["case-missing"]
    elif mutation == "nonbinary":
        metric["kind"] = "recorded"
    elif mutation == "aggregation":
        metric["aggregation"] = "median"
    elif mutation == "rounded_mean":
        metric["baseline_mean"] = 0.050001
    elif mutation == "out_of_range":
        metric["subject_mean"] = 1.1
    elif mutation == "unavailable":
        metric["baseline_mean"] = None
    elif mutation == "huge_count":
        metric["count"] = 10**400
    elif mutation == "inexact_count":
        metric["count"] = 2**53 + 1
    before = deepcopy(comparison)
    view = record_reporting._metric_views(comparison, {})[0]
    match_notes = [note for note in view.notes if note.startswith("Matches:")]
    assert match_notes == (
        [] if mutation else ["Matches: baseline 20 of 400; subject 24 of 400."]
    )
    assert comparison == before


def test_sdk_supplied_comparison_check_runs_before_shared_assembly(monkeypatch):
    snapshot, comparison, _ = binary_comparison()
    comparison["metrics"][0]["reasons"] = ["changed after binding"]

    def forbidden(*args, **kwargs):
        raise AssertionError("unbound comparison reached shared presentation")

    monkeypatch.setattr(record_reporting, "_assemble_view", forbidden)
    with pytest.raises(
        record_reporting.EvaluationRecordsError, match="differs from supplied evidence"
    ):
        record_reporting._view(comparison, snapshot)


def test_cli_scope_validation_runs_before_shared_assembly(monkeypatch):
    snapshot, _, _ = binary_comparison()
    manifest, inputs, signer = captured_reporting.load_payloads(snapshot)
    inputs["report"]["metrics"].clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("missing scopes reached shared presentation")

    monkeypatch.setattr(captured_reporting, "_assemble_view", forbidden)
    with pytest.raises(
        captured_reporting.CapturedReportError, match="contains no metrics"
    ):
        captured_reporting._view(manifest, inputs, signer)


@pytest.mark.parametrize("decision", ["regression", "insufficient_evidence"])
def test_single_metric_nonpass_summary_preserves_explanation(decision):
    _, comparison, _ = binary_comparison()
    comparison["metrics"][0].update(
        decision=decision, reasons=["Recorded requirement was not met"]
    )
    metric = record_reporting._metric_views(comparison, {})[0]
    summary = record_reporting._captured_summary((metric,))
    assert summary.startswith(metric.explanation)
    assert "Matches:" not in summary
