from __future__ import annotations

import json
from pathlib import Path

from scripts.evidence_packs.python.evaluation_optimization_summary import build_summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_summary_collects_timing_and_grouped_process_savings(tmp_path: Path):
    _write_json(
        tmp_path / "model" / "reports" / "edit" / "run_1" / "evaluate_timing.json",
        {
            "schema": "invarlock/evaluate-timing-v1",
            "baseline_report_reused": True,
            "defer_report_rendering": True,
            "timings_seconds": {
                "plan": 0.1,
                "baseline": 0.0,
                "subject": 2.0,
                "evaluation_report": 0.5,
                "total": 2.5,
            },
            "run_timings_seconds": {
                "baseline": {"load_model": 1.0, "load_dataset": 0.25, "eval": 3.0},
                "subject": {"load_model": 1.5, "load_dataset": 0.25, "eval": 4.0},
            },
        },
    )
    _write_json(
        tmp_path / "model" / "evaluation_groups" / "task_1" / "summary.json",
        {
            "schema": "invarlock/evidence-pack-evaluate-group-summary-v1",
            "completed_entries": 2,
            "avoided_cli_process_invocations": 1,
        },
    )
    _write_json(
        tmp_path / "model" / "evaluation_groups" / "task_2" / "summary.json",
        {
            "schema": "invarlock/evidence-pack-evaluate-group-summary-v1",
            "completed_entries": 2,
            "avoided_cli_process_invocations": 1,
        },
    )

    summary = build_summary(tmp_path)

    assert summary["evaluation_reports_timed"] == 1
    assert summary["baseline_report_reuse_count"] == 1
    assert summary["deferred_rendering_count"] == 1
    assert summary["grouped_evaluation_tasks"] == 2
    assert summary["grouped_evaluation_entries"] == 4
    assert summary["grouped_evaluation_task_sizes"] == [2, 2]
    assert summary["grouped_evaluation_max_entries_per_task"] == 2
    assert summary["avoided_cli_process_invocations"] == 2
    assert summary["timing_totals_seconds"]["plan"] == 0.1
    assert summary["timing_totals_seconds"]["subject"] == 2.0
    assert summary["run_timing_totals_seconds"]["load_model"] == 2.5
    assert summary["run_timing_totals_seconds"]["load_dataset"] == 0.5
    assert summary["run_timing_totals_seconds"]["eval"] == 7.0
