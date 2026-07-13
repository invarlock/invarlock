from __future__ import annotations

import pytest

from invarlock.core.exceptions import ValidationError
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_types import RunReport, create_empty_report
from invarlock.reporting.run_report_formatters import to_html
from tests.reporting._support_canonical_reports import canonical_run_report


def _mk_report() -> RunReport:
    r = create_empty_report()
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["meta"]["auto"] = {
        "tier": "balanced",
        "probes_used": 0,
        "target_pm_ratio": None,
    }
    r["context"] = {"profile": "dev"}
    r["edit"]["name"] = "quant_rtn"
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    r["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
    }
    # optional metrics to ensure tables render
    r["metrics"]["latency_ms_per_tok"] = 1.23
    r["metrics"]["memory_mb_peak"] = 12.3
    return canonical_run_report(r)


def test_baseline_v1_missing_pm_final_rejects() -> None:
    report = _mk_report()
    base_v1_bad = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {}},
    }
    with pytest.raises(ValidationError, match="Baseline normalization failed"):
        make_report(report, base_v1_bad)


def test_baseline_rejects_non_dict() -> None:
    with pytest.raises(ValidationError, match="Baseline normalization failed"):
        make_report(_mk_report(), None)


def test_single_html_renders_bullet_items() -> None:
    rp = _mk_report()
    html = to_html(rp, include_css=False)
    # Executive summary renders list items
    assert "<li>" in html
