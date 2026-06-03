from __future__ import annotations

import pytest

from invarlock.core.exceptions import ValidationError
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.run_report_formatters import to_html


def _mk_report(pm_kind: str = "ppl_causal", pm_final: float = 10.0) -> dict:
    r = create_empty_report()
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["edit"]["name"] = "quant_rtn"
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    r["metrics"]["primary_metric"] = {"kind": pm_kind, "final": pm_final}
    r["metrics"]["latency_ms_per_tok"] = 1.23
    r["metrics"]["memory_mb_peak"] = 12.3
    return r


def test_make_report_bad_baseline_raises() -> None:
    rep = _mk_report()
    base = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {}},
    }
    with pytest.raises(ValidationError, match="Baseline normalization failed"):
        make_report(rep, base)


def test_comparison_html_one_side_has_guards() -> None:
    r1 = _mk_report()
    r2 = _mk_report()
    r1["guards"] = []
    r2["guards"] = [
        {
            "name": "variance",
            "policy": {},
            "metrics": {},
            "actions": [],
            "violations": ["cap"],
        }
    ]
    html = to_html(r1, compare=r2, include_css=False)
    assert "Guard Reports" in html
