from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.core.exceptions import ValidationError
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_types import RunReport, create_empty_report
from invarlock.reporting.run_report_formatters import to_json
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
        "preview": 10.0,
        "final": 10.0,
    }
    return r


def test_make_report_baseline_variants() -> None:
    report = _mk_report()
    base_ok = _mk_report()
    base_ok["edit"]["name"] = "noop"
    assert (
        make_report(canonical_run_report(report), canonical_baseline(base_ok))[
            "schema_version"
        ]
        == "v1"
    )
    base_bad = {"schema_version": "baseline-v1", "meta": {}, "metrics": {}}
    with pytest.raises(ValidationError, match="Baseline normalization failed"):
        make_report(canonical_run_report(report), base_bad)


def test_to_json_sanitizes_non_serializable(tmp_path: Path) -> None:
    rp = _mk_report()
    # Inject a non-serializable object; sanitizer should render it as string
    rp["meta"]["extra"] = (tmp_path / "unit").resolve()
    txt = to_json(rp)
    assert "unit" in txt
