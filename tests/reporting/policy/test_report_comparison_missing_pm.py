from __future__ import annotations

import pytest

from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.run_report_formatters import to_html, to_markdown


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
        r["metrics"]["primary_metric"] = {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
        }
    else:
        r["metrics"]["primary_metric"] = {"kind": "ppl_causal"}
    r["metrics"]["latency_ms_per_tok"] = 1.5
    r["metrics"]["memory_mb_peak"] = 32.0
    return r


def test_comparison_markdown_omits_primary_metric_row_when_missing() -> None:
    r1 = _mk_report(with_pm=True)
    r2 = _mk_report(with_pm=False)
    with pytest.raises(ValueError, match="Invalid comparison RunReport structure"):
        to_markdown(r1, compare=r2)
    with pytest.raises(ValueError, match="Invalid comparison RunReport structure"):
        to_html(r1, compare=r2, include_css=False)
