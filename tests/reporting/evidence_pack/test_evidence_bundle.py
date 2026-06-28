from __future__ import annotations

import os
from pathlib import Path

from invarlock.reporting.report_bundle import save_evaluation_bundle
from invarlock.reporting.report_make import make_report


def _minimal_run_report() -> dict:
    return {
        "meta": {
            "model_id": "stub",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 7,
        },
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "x",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
        "flags": {"guard_recovered": False, "rollback_reason": None},
        "metrics": {
            # Provide PM to satisfy validate_report used by save_report
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            },
            # Legacy fields can still be present
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "bootstrap": {"replicates": 200, "alpha": 0.05, "method": "percentile"},
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [2.3]}},
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def test_manifest_and_expected_files_present(tmp_path: Path) -> None:
    primary = _minimal_run_report()
    baseline = _minimal_run_report()
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use save_report to emit evaluation report + manifest; enable evidence hook
    os.environ["INVARLOCK_EVIDENCE_DEBUG"] = "1"
    try:
        _ = save_evaluation_bundle(
            run_report=primary,
            output_dir=out_dir,
            evaluation_report=make_report(primary, baseline),
        )
    finally:
        os.environ.pop("INVARLOCK_EVIDENCE_DEBUG", None)

    assert (out_dir / "evaluation.report.json").exists(), (
        "evaluation report JSON missing"
    )
    assert (out_dir / "evaluation_report.md").exists(), (
        "evaluation report Markdown missing"
    )
    assert (out_dir / "evidence_summary.txt").exists(), "evidence summary missing"
    # Evidence file should be present when debug flag is on
    ev_file = out_dir / "guards_evidence.json"
    assert ev_file.exists(), "guards_evidence.json missing"
    assert ev_file.stat().st_size < 100 * 1024, "evidence payload too large"
