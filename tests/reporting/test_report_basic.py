from pathlib import Path
from typing import Any

from invarlock.reporting import report as rpt
from invarlock.reporting.report import (
    save_report,
    to_certificate,
    to_html,
    to_json,
    to_markdown,
)


def _minimal_report() -> dict[str, Any]:
    return {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "commit": "abc",
            "seed": 1,
            "device": "cpu",
            "ts": "2024-01-01T00:00:00",
            "auto": None,
        },
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 4,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "deadbeef",
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "layers_modified": 0,
            },
        },
        "guards": [],
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 100.0},
            "latency_ms_per_tok": 1.23,
            "memory_mb_peak": 256.0,
        },
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_to_json_markdown_html_roundtrip():
    rep = _minimal_report()
    j = to_json(rep)
    assert '"model_id": "m"' in j

    md = to_markdown(rep)
    assert "InvarLock Evaluation Report" in md

    html = to_html(rep, include_css=False)
    assert "<html" in html and "InvarLock Evaluation Report" in html

    # comparison path
    md_cmp = to_markdown(rep, compare=rep)
    assert "Comparison" in md_cmp
    html_cmp = to_html(rep, compare=rep, include_css=False)
    assert "Comparison" in html_cmp


def test_save_report_multiple_formats(tmp_path: Path):
    rep = _minimal_report()
    out = save_report(
        rep, tmp_path, formats=["json", "markdown", "html"], filename_prefix="r"
    )
    assert out["json"].exists()
    assert out["markdown"].exists()
    assert out["html"].exists()


def test_to_certificate_markdown_path():
    rep = _minimal_report()
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 100.0}},
    }
    md = to_certificate(rep, baseline, format="markdown")
    assert isinstance(md, str) and "InvarLock Safety Certificate" in md


def test_validate_baseline_or_report_helper():
    # Valid as RunReport
    assert rpt._validate_baseline_or_report(_minimal_report()) is True
    # Valid as baseline-v1 with primary_metric.final
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 100.0}},
    }
    assert rpt._validate_baseline_or_report(baseline) is True
