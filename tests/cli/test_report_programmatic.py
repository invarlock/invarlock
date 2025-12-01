from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli.commands import report as R
from invarlock.reporting.report_types import create_empty_report


def test_report_command_programmatic_json_only(tmp_path: Path):
    # Minimal run report structure for loader
    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report = create_empty_report()
    report["meta"]["model_id"] = "test"
    report["meta"]["adapter"] = "hf_gpt2"
    report["metrics"]["ppl_ratio"] = 1.0
    report["metrics"]["ppl_final"] = 1.0
    report["metrics"]["ppl_preview"] = 1.0
    report["metrics"]["ppl_preview_ci"] = (1.0, 1.0)
    report["metrics"]["ppl_final_ci"] = (1.0, 1.0)
    report["metrics"]["ppl_ratio_ci"] = (1.0, 1.0)
    # PM-only: include minimal primary_metric to satisfy validation used by to_json
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 1.0,
        "final": 1.0,
        "ratio_vs_baseline": 1.0,
        "display_ci": [1.0, 1.0],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    # Should generate plain JSON report bundle without baseline
    R.report_command(
        run=str(run_dir),
        format="json",
        baseline=None,
        compare=None,
        output=str(tmp_path / "out"),
    )

    out = tmp_path / "out"
    assert out.exists()
    files = list(out.glob("*.json"))
    assert files, "Expected at least one JSON artifact"
