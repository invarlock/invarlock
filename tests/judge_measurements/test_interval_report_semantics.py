"""Judge chart geometry follows the bound policy without replacing its result."""

from html import unescape

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


@pytest.mark.parametrize("direction", ["higher", "lower"])
@pytest.mark.parametrize("role", ["required", "advisory"])
@pytest.mark.parametrize(
    "outcome", ["pass", "regression", "crossing", "precision", "units", "incomplete"]
)
def test_numeric_policy_checks_in_html_markdown_and_terminal(
    tmp_path, direction, role, outcome
):
    from typer.testing import CliRunner

    from invarlock.cli.app import app
    from invarlock.judge_measurements.reporting import render_judge_evidence

    good = 1 if direction == "higher" else 0
    score = 1 - good if outcome == "regression" else good
    changes = {
        "direction": direction,
        "allowed_degradation": "1",
        "subject_bound": "0.731" if direction == "higher" else "0.269",
    }
    if outcome == "pass":
        changes["subject_bound"] = "0.5"
    if outcome == "precision":
        changes["maximum_interval_width"] = "0.1"
    if outcome == "units":
        changes["minimum_units"] = 100
    publication, _ = _publish(
        tmp_path,
        baseline=score,
        subject=score,
        role=role,
        incomplete=outcome == "incomplete",
        policy_changes=changes,
    )
    html = tmp_path / "report.html"
    md = tmp_path / "report.md"
    result = render_judge_evidence(publication.path, html_path=html, markdown_path=md)
    cli = CliRunner().invoke(app, ["report", str(publication.path)])
    assert cli.exit_code == 0, cli.output
    retained, artifacts = _snapshot(publication.path)
    view, facts = _view(retained, artifacts)
    assert facts["analysis"] == publication.analysis_result.to_dict()
    checks = {check.name: check for check in view.metrics[0].checks}
    assert checks["Schedule completeness"].observed == (
        "31/32 completed trials"
        if outcome == "incomplete"
        else "32/32 completed trials"
    )
    assert checks["Schedule completeness"].passed is (
        None if outcome == "incomplete" else True
    )
    assert checks["Independent units"].observed == (
        "15 complete / 16 scheduled"
        if outcome == "incomplete"
        else "16 complete / 16 scheduled"
    )
    assert checks["Independent units"].passed is (
        None if outcome in {"units", "incomplete"} else True
    )
    gate = checks["subject_bound"]
    assert gate.passed is (
        True if outcome == "pass" else False if outcome == "regression" else None
    )
    for rendered in (result.text, md.read_text(), html.read_text()):
        text = unescape(rendered)
        assert f"Decision role: {role}" in text
        assert "Schedule completeness" in text
        assert checks["Schedule completeness"].observed in text
        assert "fixed-benchmark-hoeffding-v1" in text
        assert "family confidence at least 95%" in text
        assert "alpha 0.05; comparison family size 2" in text
        assert "subject_bound precision" in text
        assert gate.required in text
        assert gate.observed in text
        assert checks["paired_effect"].required in text
        assert checks["paired_effect"].observed in text
        assert checks["paired_effect precision"].observed in text
        assert checks["paired_effect precision"].required in text
        if outcome == "incomplete":
            assert "Unavailable; incomplete planned schedule" in text
        else:
            interval = facts["analysis"]["subject_interval"]
            assert f"lower {interval['lower']}; upper {interval['upper']}" in text
            adverse_endpoint = "upper <" if direction == "higher" else "lower >"
            assert f"{adverse_endpoint} {changes['subject_bound']}" in text
        if outcome == "regression":
            assert "adverse" in text
            # The original fixture's decisive bound must appear in the primary report.
            if direction == "higher":
                assert "upper 0.370051796825200" in text
        elif outcome != "pass":
            assert "inconclusive" in text

    terminal = " ".join(cli.stdout.split())
    assert "family confidence at least 95%" in terminal
    assert "fixed-benchmark-hoeffding-v1" in terminal
    assert changes["subject_bound"] in terminal
    assert "paired_effect" in terminal and "subject_bound" in terminal
    assert (
        "inconclusive" in terminal
        if outcome not in {"pass", "regression"}
        else "satisfied" in terminal
    )
    if outcome != "incomplete":
        assert facts["analysis"]["subject_interval"]["upper"] in terminal
        assert facts["analysis"]["subject_interval"]["lower"] in terminal
    else:
        assert "31/32" in terminal and "Unavailable" in terminal


def test_unconfigured_subject_bound_remains_descriptive(tmp_path):
    from invarlock.judge_measurements.reporting import render_judge_evidence

    publication, _ = _publish(tmp_path)
    result = render_judge_evidence(publication.path)
    retained, artifacts = _snapshot(publication.path)
    view, _ = _view(retained, artifacts)
    names = {check.name for check in view.metrics[0].checks}
    assert "subject_bound" not in names
    assert "subject_bound precision" not in names
    assert "Subject interval (descriptive; no subject bound configured)" in unescape(
        result.text
    )


@pytest.mark.parametrize("direction", ["higher", "lower"])
@pytest.mark.parametrize("outcome", ["pass", "regression", "insufficient_evidence"])
def test_effect_gate_reports_signed_threshold_and_recorded_outcome(
    tmp_path, direction, outcome
):
    from invarlock.judge_measurements.reporting import render_judge_evidence

    baseline, subject = (0, 1) if direction == "higher" else (1, 0)
    if outcome == "regression":
        baseline, subject = subject, baseline
    elif outcome == "insufficient_evidence":
        baseline = subject
    publication, _ = _publish(
        tmp_path,
        baseline=baseline,
        subject=subject,
        policy_changes={"direction": direction, "allowed_degradation": "0.25"},
    )
    result = render_judge_evidence(publication.path)
    retained, artifacts = _snapshot(publication.path)
    view, _ = _view(retained, artifacts)
    check = next(c for c in view.metrics[0].checks if c.name == "paired_effect")
    assert result.facts["analysis"]["decision"] == outcome
    assert (
        check.passed
        is {"pass": True, "regression": False, "insufficient_evidence": None}[outcome]
    )
    assert check.required == (
        "lower >= -0.25; adverse if upper < -0.25 (required)"
        if direction == "higher"
        else "upper <= 0.25; adverse if lower > 0.25 (required)"
    )
    assert check.required in unescape(result.text)
    assert check.observed in unescape(result.text)
