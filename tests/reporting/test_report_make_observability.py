from __future__ import annotations

from copy import deepcopy

from invarlock.reporting.report_make import make_report
from tests.reporting.test_report_full_context import _rich_run_report


def test_make_evaluation_report_marks_tiny_relax() -> None:
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    report["context"] = {"run": {"tiny_relax": True}}
    evaluation_report = make_report(report, baseline)
    assert evaluation_report["auto"]["tiny_relax"] is True
    stats = evaluation_report["dataset"]["windows"]["stats"]
    assert "coverage" in stats and "window_match_fraction" in stats
    qo = evaluation_report.get("quality_overhead")
    if qo:
        assert qo["basis"] in {"ratio", "delta_pp"}


def test_make_evaluation_report_embeds_telemetry_summary(monkeypatch):
    report, baseline = _rich_run_report()
    report = deepcopy(report)
    baseline = deepcopy(baseline)
    monkeypatch.setenv("INVARLOCK_TELEMETRY", "1")
    evaluation_report = make_report(report, baseline)
    assert evaluation_report["telemetry"]["summary_line"].startswith(
        "INVARLOCK_TELEMETRY"
    )
