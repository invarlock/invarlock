from __future__ import annotations

import json

import pytest
from invarlock.reporting.render import render_report_markdown

from invarlock.core.exceptions import ValidationError
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_types import RunReport, create_empty_report


def _mk_report() -> RunReport:
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
    return r


def test_make_report_json_and_markdown() -> None:
    rp = _mk_report()
    base = _mk_report()
    cert = make_report(rp, base)
    js = json.dumps(cert, indent=2, ensure_ascii=False)
    assert "schema_version" in js
    md = render_report_markdown(cert)
    assert "Evaluation Report" in md


def test_make_report_rejects_unsupported_baseline_schema() -> None:
    rp = _mk_report()
    base = {
        "schema_version": "baseline-v2",
        "meta": {"model_id": "m"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    with pytest.raises(ValidationError, match="Baseline normalization failed"):
        make_report(rp, base)
