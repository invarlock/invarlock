from __future__ import annotations

from invarlock.reporting.report import to_html, to_markdown
from invarlock.reporting.report_types import create_empty_report


def _mk_report(with_pm: bool = True) -> dict:
    r = create_empty_report()
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["edit"]["name"] = "quant_rtn"
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    if with_pm:
        r["metrics"]["primary_metric"] = {"kind": "ppl_causal", "final": 10.0}
    else:
        r["metrics"]["primary_metric"] = {"kind": "ppl_causal"}
    r["metrics"]["latency_ms_per_tok"] = 1.5
    r["metrics"]["memory_mb_peak"] = 32.0
    return r


def test_comparison_markdown_omits_primary_metric_row_when_missing() -> None:
    r1 = _mk_report(with_pm=True)
    r2 = _mk_report(with_pm=False)
    md = to_markdown(r1, compare=r2)
    # Ensure table exists and no Primary Metric row when one side is missing
    assert "Comparison Summary" in md
    assert "Primary Metric" not in md.split("\n\n")[1]  # in first table block
    html = to_html(r1, compare=r2, include_css=False)
    assert "<table" in html
