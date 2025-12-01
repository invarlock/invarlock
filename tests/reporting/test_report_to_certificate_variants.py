from __future__ import annotations

import pytest

from invarlock.reporting.report import to_certificate
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


def test_to_certificate_json_and_markdown(tmp_path) -> None:
    rp = _mk_report()
    base = _mk_report()
    js = to_certificate(rp, base, format="json")
    assert "schema_version" in js
    md = to_certificate(rp, base, format="markdown")
    assert "Safety Certificate" in md


def test_to_certificate_unsupported_format(tmp_path) -> None:
    rp = _mk_report()
    base = _mk_report()
    with pytest.raises(ValueError):
        to_certificate(rp, base, format="txt")
