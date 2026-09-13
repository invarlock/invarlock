"""Judge chart geometry follows the bound policy without replacing its result."""

import pytest

from invarlock.judge_measurements.reporting import _snapshot, _view
from tests.judge_measurements.test_evidence_acceptance import _publish


@pytest.mark.parametrize(
    "direction,threshold_direction,threshold",
    [("higher", "minimum", -0.25), ("lower", "maximum", 0.25)],
)
def test_judge_interval_has_explicit_policy_direction_and_neutral(
    tmp_path, direction, threshold_direction, threshold
):
    publication, _ = _publish(
        tmp_path,
        baseline=0,
        subject=0,
        policy_changes={
            "direction": direction,
            "allowed_degradation": "0.25",
            "minimum_units": 100,
        },
    )
    retained, artifacts = _snapshot(publication.path)
    view, _ = _view(retained, artifacts)
    metric = view.metrics[0]
    assert metric.interval.threshold_direction == threshold_direction
    assert metric.interval.threshold == threshold
    assert metric.interval.neutral == 0.0
    assert metric.interval.estimate == 0.0
    assert (
        metric.decision
        == publication.analysis_result.decision
        == "insufficient_evidence"
    )
    assert any(check.passed is None for check in metric.checks)


def test_judge_display_compacts_numbers_without_changing_retained_precision(tmp_path):
    publication, _ = _publish(tmp_path, baseline=0, subject=1)
    retained, artifacts = _snapshot(publication.path)
    view, facts = _view(retained, artifacts)
    metric = view.metrics[0]
    assert (metric.baseline, metric.candidate, metric.change) == ("0", "1", "1")
    assert metric.count == "16 cases"
    assert "16 independent units; 32/32 completed trials." in metric.notes
    assert facts["descriptive_means"]["baseline"] == "0E-15"
