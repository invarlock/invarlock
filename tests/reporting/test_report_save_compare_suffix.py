from __future__ import annotations

from pathlib import Path

from invarlock.reporting.report import save_report
from invarlock.reporting.report_types import create_empty_report


def _mk_report() -> dict:
    r = create_empty_report()
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["edit"]["name"] = "quant_rtn"
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    r["metrics"]["primary_metric"] = {"kind": "ppl_causal", "final": 10.0}
    r["metrics"]["latency_ms_per_tok"] = 1.0
    r["metrics"]["memory_mb_peak"] = 10.0
    return r


def test_save_report_with_compare_writes_comparison_suffix(tmp_path: Path) -> None:
    r1 = _mk_report()
    r2 = _mk_report()
    files = save_report(
        r1,
        tmp_path,
        formats=["json", "markdown", "html"],
        compare=r2,
        filename_prefix="rep",
    )
    assert files["json"].name.endswith("_comparison.json")
    assert files["markdown"].name.endswith("_comparison.md")
    assert files["html"].name.endswith("_comparison.html")
