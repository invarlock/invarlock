from __future__ import annotations

import copy
import json
from pathlib import Path

from invarlock.reporting.rendering.markdown import render_report_markdown
from invarlock.reporting.report_bundle import save_evaluation_bundle, save_report
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_types import RunReport
from invarlock.reporting.run_report_formatters import to_html, to_json, to_markdown
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def make_min_report() -> RunReport:
    return canonical_run_report(
        RunReport(
            meta={
                "model_id": "m",
                "adapter": "hf",
                "commit": "deadbeef",
                "seed": 1,
                "device": "cpu",
                "ts": "2025-01-01T00:00:00",
                "auto": {
                    "tier": "balanced",
                    "probes_used": 0,
                    "target_pm_ratio": None,
                },
            },
            context={"profile": "dev"},
            data={
                "dataset": "ds",
                "split": "val",
                "seq_len": 4,
                "stride": 4,
                "preview_n": 1,
                "final_n": 1,
            },
            edit={
                "name": "quant_rtn",
                "plan_digest": "abc123",
                "deltas": {
                    "params_changed": 1,
                    "sparsity": None,
                    "bitwidth_map": None,
                    "layers_modified": 1,
                },
            },
            guards=[],
            metrics={
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": 10.0,
                    "final": 10.0,
                },
                "preview_total_tokens": 100,
                "final_total_tokens": 100,
                "latency_ms_per_tok": 1.23,
                "memory_mb_peak": 256.0,
            },
            artifacts={"events_path": "", "logs_path": "", "checkpoint_path": None},
            flags={"guard_recovered": False, "rollback_reason": None},
        )
    )


def make_baseline() -> dict:
    baseline = copy.deepcopy(make_min_report())
    baseline["meta"]["commit"] = "beefdead"
    baseline["edit"]["name"] = "noop"
    baseline["edit"]["deltas"]["params_changed"] = 0
    return canonical_baseline(baseline)


def test_to_json_markdown_html_variants(tmp_path: Path) -> None:
    report = make_min_report()
    # JSON
    js = to_json(report)
    obj = json.loads(js)
    assert obj["meta"]["model_id"] == "m"
    # Markdown single and compare (dict path)
    md1 = to_markdown(report)
    assert "InvarLock Evaluation Report" in md1
    md2 = to_markdown(report, compare=make_min_report())
    assert "Comparison" in md2
    # HTML with and without CSS
    html1 = to_html(report, include_css=True)
    assert "<html" in html1 and "<style" in html1
    html2 = to_html(report, include_css=False)
    assert "<style" not in html2


def test_make_report_and_save_report(tmp_path: Path, monkeypatch) -> None:
    report = make_min_report()
    base = make_baseline()

    cert = make_report(report, base)
    report_json = json.dumps(cert, indent=2, ensure_ascii=False)
    assert json.loads(report_json)["schema_version"]
    report_md = render_report_markdown(cert)
    assert "Evaluation Report" in report_md

    # save_report without baseline for report should error
    out = tmp_path / "out"
    import pytest

    with pytest.raises(ValueError, match="save_evaluation_bundle"):
        save_report(report, out, formats=["report"])

    # Enable evidence emission
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")
    save_report(report, out, formats=["json", "markdown", "html"])
    save_evaluation_bundle(
        run_report=report,
        output_dir=out,
        evaluation_report=make_report(report, base),
    )
    # Basic outputs exist
    assert (out / "report.json").exists()
    assert (out / "report.md").exists()
    assert (out / "report.html").exists()
    # Evaluation report artifacts
    assert (out / "evaluation.report.json").exists()
    assert (out / "evaluation_report.md").exists()
    # Manifest present and references evidence when env enabled
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "files" in manifest
    ev_path = out / "guards_evidence.json"
    assert ev_path.exists()
    assert "evidence" in manifest
    summary = manifest.get("summary", {})
    assert summary.get("overall_status") in {"PASS", "FAIL"}
    assert isinstance(summary.get("gates_passed"), int)
    assert isinstance(summary.get("gates_total"), int)
    assert summary.get("primary_metric_ratio") is None or isinstance(
        summary.get("primary_metric_ratio"), float
    )
