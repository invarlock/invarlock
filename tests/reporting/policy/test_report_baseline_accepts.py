from __future__ import annotations

import pytest

from invarlock.core.exceptions import ValidationError
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_types import RunReport, create_empty_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


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


def test_make_report_accepts_canonical_run_report_baseline() -> None:
    rp = _mk_report()
    baseline = _mk_report()
    baseline["edit"]["name"] = "noop"
    cert = make_report(canonical_run_report(rp), canonical_baseline(baseline))
    assert cert.get("schema_version") == "v1"

    legacy = {
        "schema_version": "baseline-v1",
        "meta": {"model_id": "m"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    with pytest.raises(ValidationError, match="Baseline normalization failed"):
        make_report(rp, legacy)


def test_make_report_rejects_invalid_baseline() -> None:
    rp = _mk_report()
    bad_base = {
        "meta": {"model_id": "m"},
        # Missing primary_metric.final makes it invalid
        "metrics": {"primary_metric": {"kind": "ppl_causal"}},
    }
    with pytest.raises(ValidationError, match="Baseline normalization failed"):
        make_report(rp, bad_base)
