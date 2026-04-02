from __future__ import annotations

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


def test_make_report_accepts_baseline_v1() -> None:
    rp = _mk_report()
    base_v1 = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    cert = make_report(rp, base_v1)
    assert cert.get("schema_version") == "v1"

    import pytest

    base_v2 = {
        "schema_version": "baseline-v2",
        "meta": {"model_id": "m"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    with pytest.raises(ValueError):
        make_report(rp, base_v2)


def test_make_report_rejects_invalid_baseline() -> None:
    rp = _mk_report()
    bad_base = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m"},
        # Missing primary_metric.final makes it invalid
        "metrics": {"primary_metric": {"kind": "ppl_causal"}},
    }
    import pytest

    with pytest.raises(ValueError):
        make_report(rp, bad_base)
