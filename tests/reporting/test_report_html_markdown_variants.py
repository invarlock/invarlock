from __future__ import annotations

from pathlib import Path

from invarlock.reporting.report import to_html, to_markdown
from invarlock.reporting.report_types import RunReport, create_empty_report


def _mk_report(pm_kind: str = "ppl_causal", pm_final: float = 10.0) -> RunReport:
    r = create_empty_report()
    # Fill minimal required fields for PM
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["edit"]["name"] = "quant_rtn"
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    r["metrics"]["primary_metric"] = {
        "kind": pm_kind,
        "final": pm_final,
    }
    # Optional metrics for table rows
    r["metrics"]["latency_ms_per_tok"] = 1.23
    r["metrics"]["memory_mb_peak"] = 12.3
    return r


def test_to_html_markdown_single_and_css_toggle(tmp_path: Path) -> None:
    rp = _mk_report()
    md = to_markdown(rp)
    assert "InvarLock Evaluation Report" in md

    # HTML with CSS
    html_with = to_html(rp, include_css=True, title="Custom Title")
    assert "<style>" in html_with
    assert "<table" in html_with

    # HTML without CSS exercises the branch that appends only body/head
    html_without = to_html(rp, include_css=False)
    assert "<style>" not in html_without
    assert "<table" in html_without


def test_to_html_comparison_table_boundaries() -> None:
    rp1 = _mk_report(pm_final=10.0)
    rp2 = _mk_report(pm_final=11.0)

    # Comparison markdown has a table that needs closing paths
    html_cmp = to_html(rp1, compare=rp2)
    # Ensure both headers and rows are present and tables closed
    assert "Comparison Summary" in html_cmp
    assert html_cmp.count("<table") >= 1
    assert html_cmp.count("</table>") >= 1


def test_to_html_comparison_guards_violations_rows() -> None:
    rp1 = _mk_report(pm_final=10.0)
    rp2 = _mk_report(pm_final=10.5)
    # Add guards with violations to trigger list-item rendering in comparison
    rp1["guards"] = [
        {
            "name": "variance",
            "policy": {},
            "metrics": {},
            "actions": [],
            "violations": ["v1"],
        }
    ]
    rp2["guards"] = [
        {
            "name": "variance",
            "policy": {},
            "metrics": {},
            "actions": [],
            "violations": ["v1b"],
        }
    ]
    html_cmp = to_html(rp1, compare=rp2)
    assert "Guard Reports" in html_cmp
