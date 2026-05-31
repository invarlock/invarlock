from pathlib import Path
from typing import Any

from invarlock.reporting.render import render_report_markdown
from invarlock.reporting.report_files import save_report

from invarlock.reporting.report_make import make_report
from invarlock.reporting.run_report_formatters import (
    _generate_comparison_markdown,
    _generate_single_markdown,
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


def test_make_report_markdown_path():
    rep = _minimal_report()
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 100.0}},
    }
    md = render_report_markdown(make_report(rep, baseline))
    assert isinstance(md, str) and "InvarLock Evaluation Report" in md


def test_make_report_accepts_run_report_and_baseline_v1():
    rep = _minimal_report()
    cert_from_run = make_report(rep, rep)
    assert cert_from_run["schema_version"] == "v1"

    baseline = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 100.0}},
    }
    cert_from_baseline = make_report(rep, baseline)
    assert cert_from_baseline["schema_version"] == "v1"


def test_single_markdown_handles_missing_primary_metric_and_sparse_guards():
    rep = _minimal_report()
    rep["metrics"]["primary_metric"] = {}
    rep["edit"]["deltas"]["sparsity"] = 0.25
    rep["guards"] = [
        {"name": "variance", "metrics": {}, "actions": [], "violations": []}
    ]
    rep["flags"]["guard_recovered"] = True

    md = "\n".join(_generate_single_markdown(rep))

    assert "Primary Metric**: unavailable" in md
    assert "Overall Sparsity" in md
    assert "Guard recovery was triggered" in md


def test_comparison_markdown_coerces_invalid_delta_values():
    rep1 = _minimal_report()
    rep2 = _minimal_report()
    rep1["metrics"]["primary_metric"] = {}
    rep2["metrics"]["primary_metric"] = {}
    rep1["edit"]["deltas"]["params_changed"] = "bad"
    rep2["edit"]["deltas"]["layers_modified"] = "bad"
    rep1["guards"] = [
        {"name": "spectral", "violations": [], "metrics": {}, "actions": []}
    ]
    rep2["guards"] = [
        {"name": "spectral", "violations": ["x"], "metrics": {}, "actions": []}
    ]

    md = "\n".join(_generate_comparison_markdown(rep1, rep2))

    assert "| Params Changed | 0 | 0 | +0 |" in md
    assert "| Layers Modified | 0 | 0 | +0 |" in md
    assert "Violations:" in md
