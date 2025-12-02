from __future__ import annotations

import json
from pathlib import Path

from invarlock.reporting.report import (
    save_report,
    to_certificate,
    to_html,
    to_json,
    to_markdown,
)
from invarlock.reporting.report_types import RunReport


def make_min_report() -> RunReport:
    return RunReport(
        meta={
            "model_id": "m",
            "adapter": "hf",
            "commit": "deadbeef",
            "seed": 1,
            "device": "cpu",
            "ts": "2025-01-01T00:00:00",
            "auto": None,
        },
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
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
            "preview_total_tokens": 100,
            "final_total_tokens": 100,
            "latency_ms_per_tok": 1.23,
            "memory_mb_peak": 256.0,
        },
        artifacts={"events_path": "", "logs_path": "", "checkpoint_path": None},
        flags={"guard_recovered": False, "rollback_reason": None},
    )


def make_baseline() -> dict:
    return {
        "schema_version": "baseline-v1",
        "meta": {"commit": "beefdead"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }


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


def test_to_certificate_and_save_report(tmp_path: Path, monkeypatch) -> None:
    report = make_min_report()
    base = make_baseline()

    # to_certificate supports json and markdown
    cert_json = to_certificate(report, base, format="json")
    assert json.loads(cert_json)["schema_version"]
    cert_md = to_certificate(report, base, format="markdown")
    assert "Safety Certificate" in cert_md

    # save_report without baseline for cert should error
    out = tmp_path / "out"
    import pytest

    with pytest.raises(ValueError):
        save_report(report, out, formats=["cert"])  # type: ignore[arg-type]

    # Enable evidence emission
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")
    save_report(
        report, out, formats=["json", "markdown", "html", "cert"], baseline=base
    )  # type: ignore[arg-type]
    # Basic outputs exist
    assert (out / "report.json").exists()
    assert (out / "report.md").exists()
    assert (out / "report.html").exists()
    # Certificate artifacts
    assert (out / "evaluation.cert.json").exists()
    assert (out / "report_certificate.md").exists()
    # Manifest present and references evidence when env enabled
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "files" in manifest
    ev_path = out / "guards_evidence.json"
    assert ev_path.exists()
    assert "evidence" in manifest
