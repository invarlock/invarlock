import json
from datetime import datetime
from pathlib import Path

import pytest

from invarlock.reporting.report import (
    _sanitize_for_json,
    _validate_baseline_or_report,
    save_report,
    to_certificate,
    to_html,
    to_json,
    to_markdown,
)
from invarlock.reporting.report_types import create_empty_report


def _minimal_report() -> dict:
    rep = create_empty_report()
    rep["meta"].update(
        {
            "model_id": "gpt2",
            "adapter": "hf_gpt2",
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


def test_certificate_json_and_markdown_and_save(tmp_path: Path):
    rep = _minimal_report()
    # baseline accepted as RunReport
    cert_json = to_certificate(rep, rep, format="json")
    assert json.loads(cert_json)
    cert_md = to_certificate(rep, rep, format="markdown")
    assert isinstance(cert_md, str) and len(cert_md) > 0

    # save_report writes files for multiple formats
    out = save_report(
        rep, tmp_path, formats=["json", "markdown", "html", "cert"], baseline=rep
    )
    assert {"json", "markdown", "html", "cert", "cert_md"}.issubset(out.keys())

    # cert without baseline raises
    with pytest.raises(ValueError):
        save_report(rep, tmp_path, formats=["cert"], baseline=None)


def test_validate_baseline_or_report_variants():
    rep = _minimal_report()
    # Valid as RunReport
    assert _validate_baseline_or_report(rep) is True
    # Valid baseline v1
    baseline = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.1}
        },
    }
    assert _validate_baseline_or_report(baseline) is True
    # Invalid baseline
    assert (
        _validate_baseline_or_report({"schema_version": "baseline-v1", "meta": {}})
        is False
    )
