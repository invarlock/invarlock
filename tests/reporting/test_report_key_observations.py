from __future__ import annotations

from invarlock.reporting.report import to_markdown
from invarlock.reporting.report_types import create_empty_report


def _mk(pm_ratio: float) -> dict:
    r = create_empty_report()
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["edit"]["name"] = "quant_rtn"
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    r["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "final": 10.0,
        "ratio_vs_baseline": pm_ratio,
    }
    r["metrics"]["latency_ms_per_tok"] = 1.0
    r["metrics"]["memory_mb_peak"] = 2.0
    r["edit"]["deltas"]["params_changed"] = 10
    return r


def test_key_observations_impact_categories() -> None:
    md_min = to_markdown(_mk(1.03))
    assert "minimal performance impact" in md_min.lower()
    md_mod = to_markdown(_mk(1.07))
    assert "moderate performance impact" in md_mod.lower()
    md_sig = to_markdown(_mk(1.12))
    assert "significant performance impact" in md_sig.lower()
