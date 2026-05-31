import json
from datetime import datetime
from pathlib import Path

import pytest
from invarlock.reporting.render import render_report_markdown
from invarlock.reporting.report_files import save_report

from invarlock.core.exceptions import ValidationError
from invarlock.reporting.report_bundle import save_evaluation_bundle
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.run_report_formatters import (
    _sanitize_for_json,
    to_html,
    to_json,
    to_markdown,
)


def _minimal_report() -> dict:
    rep = create_empty_report()
    rep["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "commit": "deadbeefcafebabe",
            "ts": datetime.now().isoformat(),
        }
    )
    rep["data"].update(
        {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 16,
            "stride": 8,
            "preview_n": 2,
            "final_n": 2,
        }
    )
    rep["edit"]["name"] = "structured"
    rep["edit"]["plan_digest"] = "abcd" * 8
    rep["edit"]["deltas"].update(
        {
            "params_changed": 10,
            "heads_pruned": 0,
            "neurons_pruned": 0,
            "layers_modified": 1,
            "sparsity": None,
        }
    )
    rep["metrics"].update(
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.2,
                "ratio_vs_baseline": 1.02,
                "display_ci": (10.2, 10.2),
            },
            # ppl_* fields may still be present in run report, not in cert
            "ppl_preview": 10.0,
            "ppl_final": 10.2,
            "ppl_ratio": 1.02,
            "latency_ms_per_tok": 1.5,
            "memory_mb_peak": 123.4,
        }
    )
    rep["guards"] = [
        {
            "name": "invariants",
            "policy": {"strict": True},
            "metrics": {"count": 3},
            "actions": ["check"],
            "violations": ["rule_1"],
        },
        {
            "name": "spectral",
            "policy": {"sigma_quantile": 0.95},
            "metrics": {"max_sigma": 3.2},
            "actions": ["cap"],
            "violations": [],
        },
    ]
    rep["artifacts"].update({"events_path": "events.jsonl", "logs_path": "out.log"})
    rep["flags"].update({"guard_recovered": False, "rollback_reason": None})
    return rep


def test_to_json_and_sanitize():
    rep = _minimal_report()
    js = to_json(rep)
    payload = json.loads(js)
    assert payload["meta"]["model_id"] == "gpt2"
    # sanitize helper converts datetimes and unknown objects
    out = _sanitize_for_json({"now": datetime.now(), "obj": object()})
    assert isinstance(out["now"], str) and isinstance(out["obj"], str)


def test_to_markdown_and_html_single_and_compare():
    rep1 = _minimal_report()
    rep2 = _minimal_report()
    md1 = to_markdown(rep1)
    md2 = to_markdown(rep1, compare=rep2, title="Comparison")
    assert "InvarLock Evaluation Report" in md1
    assert "Comparison" in md2

    html1 = to_html(rep1, include_css=False)
    html2 = to_html(rep1, compare=rep2, title="Compare HTML", include_css=True)
    assert "<!DOCTYPE html>" in html1
    assert "Compare HTML" in html2


def test_evaluation_report_json_and_markdown_and_save(tmp_path: Path):
    rep = _minimal_report()
    cert = make_report(rep, rep)
    report_json = json.dumps(cert, indent=2, ensure_ascii=False)
    assert json.loads(report_json)
    report_md = render_report_markdown(cert)
    assert isinstance(report_md, str) and len(report_md) > 0

    # save_report writes files for multiple formats
    out = save_report(rep, tmp_path, formats=["json", "markdown", "html"])
    out.update(
        save_evaluation_bundle(
            run_report=rep,
            output_dir=tmp_path,
            evaluation_report=cert,
        )
    )
    assert {"json", "markdown", "html", "report", "report_md"}.issubset(out.keys())

    with pytest.raises(ValueError, match="save_evaluation_bundle"):
        save_report(rep, tmp_path, formats=["report"])


def test_make_report_accepts_run_report_and_baseline_v1():
    rep = _minimal_report()
    assert make_report(rep, rep)["schema_version"] == "v1"
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.1}
        },
    }
    assert make_report(rep, baseline)["schema_version"] == "v1"
    with pytest.raises(ValidationError, match="Baseline normalization failed"):
        make_report(rep, {"schema_version": "baseline-v1", "meta": {}})
