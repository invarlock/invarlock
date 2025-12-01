from __future__ import annotations

from datetime import datetime

import pytest

from invarlock.reporting.report import (
    _sanitize_for_json,
    _validate_baseline_or_report,
    to_html,
    to_json,
    to_markdown,
)
from invarlock.reporting.report_types import create_empty_report


def _minimal_report():
    r = create_empty_report()
    # Populate fields used in markdown/HTML rendering
    r["meta"].update(
        {
            "model_id": "demo-model",
            "adapter": "hf_gpt2",
            "commit": "deadbeefcafebabe",
            "device": "cpu",
        }
    )
    r["data"].update(
        {
            "dataset": "toyset",
            "split": "validation",
            "seq_len": 8,
            "preview_n": 2,
            "final_n": 2,
        }
    )
    r["edit"].update(
        {
            "name": "structured",
            "plan_digest": "0123456789abcdef0123",
            "deltas": {
                "params_changed": 1,
                "sparsity": None,
                "bitwidth_map": None,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        }
    )
    r["metrics"].update(
        {
            "ppl_preview": 10.0,
            "ppl_final": 9.5,
            "ppl_ratio": 0.95,
            "ppl_preview_ci": (9.8, 10.2),
            "ppl_final_ci": (9.3, 9.7),
            "ppl_ratio_ci": (0.93, 0.97),
            "latency_ms_per_tok": 1.23,
            "memory_mb_peak": 123.4,
            # PM-only: include a minimal primary_metric to satisfy validation
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 9.5,
                "ratio_vs_baseline": 0.95,
                "display_ci": [9.3, 10.2],
            },
        }
    )
    return r


def test_to_json_invalid_raises():
    with pytest.raises(ValueError):
        to_json({})  # type: ignore[arg-type]


def test_to_markdown_title_and_compare():
    r1 = _minimal_report()
    r2 = _minimal_report()
    md = to_markdown(r1, compare=r2, title="My Title")
    assert "# My Title" in md and "Comparison" not in md.splitlines()[0]


def test_to_html_without_css():
    r = _minimal_report()
    html = to_html(r, include_css=False)
    assert "<html" in html and "<style" not in html


def test_sanitize_for_json_properties():
    obj = {
        "dt": datetime(2025, 1, 2, 3, 4, 5),
        "num": 1,
        "flt": 2.0,
        "txt": "ok",
        "lst": [1, "x"],
        "tup": (2, 3),
        "obj": object(),
    }
    out = _sanitize_for_json(obj)
    assert isinstance(out["dt"], str) and out["num"] == 1 and out["lst"][1] == "x"


def test_validate_baseline_v1_schema():
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {"tokenizer_hash": "abc"},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 9.5}
        },
    }
    assert _validate_baseline_or_report(baseline) is True


def test_to_markdown_default_title():
    r = _minimal_report()
    md = to_markdown(r)
    assert md.splitlines()[0] == "# InvarLock Evaluation Report"


def test_to_html_with_compare():
    r1 = _minimal_report()
    r2 = _minimal_report()
    html_doc = to_html(r1, compare=r2)
    assert "comparison" in html_doc.lower()


def test_save_report_cert_requires_baseline(tmp_path):
    from invarlock.reporting.report import save_report

    r = _minimal_report()
    with pytest.raises(ValueError):
        save_report(r, tmp_path, formats=["cert"])  # missing baseline


def test_to_certificate_markdown_smoke():
    from invarlock.reporting.report import to_certificate

    r = _minimal_report()
    baseline = _minimal_report()
    md = to_certificate(r, baseline, format="markdown")
    assert isinstance(md, str) and len(md) > 0


def test_markdown_guard_reports_section():
    r1 = _minimal_report()
    r2 = _minimal_report()
    # add guard entries with violations to trigger guard report section
    r1["guards"] = [
        {
            "name": "spectral",
            "policy": {},
            "metrics": {},
            "actions": [],
            "violations": ["v1"],
        }
    ]
    r2["guards"] = [
        {
            "name": "spectral",
            "policy": {},
            "metrics": {},
            "actions": [],
            "violations": [],
        }
    ]
    md = to_markdown(r1, compare=r2)
    assert "Guard Reports" in md and "spectral" in md


def test_validate_baseline_invalid_returns_false():
    assert _validate_baseline_or_report({"schema_version": "x", "metrics": {}}) is False
