from __future__ import annotations

import json

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


def test_to_certificate_accepts_baseline_v1() -> None:
    rp = _mk_report()
    # Baseline v1
    base_v1 = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    js1 = to_certificate(rp, base_v1, format="json")
    obj1 = json.loads(js1)
    assert obj1.get("schema_version") == "v1"

    # Unsupported baseline version; ensure rejection
    import pytest

    base_v2 = {
        "schema_version": "baseline-v2",
        "meta": {"model_id": "m"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    with pytest.raises(ValueError):
        to_certificate(rp, base_v2, format="json")


def test_to_certificate_rejects_invalid_baseline() -> None:
    rp = _mk_report()
    bad_base = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m"},
        # Missing primary_metric.final makes it invalid
        "metrics": {"primary_metric": {"kind": "ppl_causal"}},
    }
    import pytest

    with pytest.raises(ValueError):
        to_certificate(rp, bad_base, format="json")
