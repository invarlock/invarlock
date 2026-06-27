from __future__ import annotations

import json
from pathlib import Path

from scripts.evidence_packs.python.validation_state import (
    build_evaluation_optimization_summary,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_summary_collects_timing_and_reuse_counts(tmp_path: Path):
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
    summary = build_evaluation_optimization_summary(tmp_path)

    assert summary["run_dir"] == "."
    assert summary["path_scope"] == "run_root_relative"
    assert summary["evaluation_reports_timed"] == 1
    assert summary["baseline_report_reuse_count"] == 1
    assert summary["deferred_rendering_count"] == 1
    assert summary["timing_totals_seconds"]["plan"] == 0.1
    assert summary["timing_totals_seconds"]["subject"] == 2.0
    assert summary["run_timing_totals_seconds"]["load_model"] == 2.5
    assert summary["run_timing_totals_seconds"]["load_dataset"] == 0.5
    assert summary["run_timing_totals_seconds"]["eval"] == 7.0
