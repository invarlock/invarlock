from __future__ import annotations

from invarlock.reporting.report import _validate_baseline_or_report, to_html
from invarlock.reporting.report_types import RunReport, create_empty_report


def _mk_report() -> RunReport:
    r = create_empty_report()
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["edit"]["name"] = "quant_rtn"
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    r["metrics"]["primary_metric"] = {"kind": "ppl_causal", "final": 10.0}
    # optional metrics to ensure tables render
    r["metrics"]["latency_ms_per_tok"] = 1.23
    r["metrics"]["memory_mb_peak"] = 12.3
    return r


def test_baseline_v1_missing_pm_final_rejects() -> None:
    base_v1_bad = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {}},
    }
    assert _validate_baseline_or_report(base_v1_bad) is False


def test_baseline_rejects_non_dict() -> None:
    assert _validate_baseline_or_report(None) is False


def test_single_html_renders_bullet_items() -> None:
    rp = _mk_report()
    html = to_html(rp, include_css=False)
    # Executive summary renders list items
    assert "<li>" in html
