from __future__ import annotations

from invarlock.reporting.report import to_markdown
from invarlock.reporting.report_types import create_empty_report


def _mk_report() -> dict:
    r = create_empty_report()
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    r["edit"]["name"] = "quant_rtn"
    r["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "final": 10.0,
        "preview": 10.0,
    }
    r["metrics"]["latency_ms_per_tok"] = 1.2
    r["metrics"]["memory_mb_peak"] = 42.0
    return r


def test_to_markdown_with_custom_title() -> None:
    rp = _mk_report()
    md = to_markdown(rp, title="Custom Title")
    assert md.startswith("# Custom Title")
