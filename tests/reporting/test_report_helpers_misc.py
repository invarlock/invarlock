from __future__ import annotations

from pathlib import Path

from invarlock.reporting.report import _validate_baseline_or_report, to_json
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
    r["metrics"]["primary_metric"] = {"kind": "ppl_causal", "final": 10.0}
    return r


def test_validate_baseline_or_report_variants() -> None:
    base_ok = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {"final": 10.0}},
    }
    assert _validate_baseline_or_report(base_ok) is True
    base_bad = {"schema_version": "baseline-v1", "meta": {}, "metrics": {}}
    assert _validate_baseline_or_report(base_bad) is False
    base_v1 = {
        "schema_version": "baseline-v1",
        "meta": {},
        "metrics": {"primary_metric": {"final": 5.0}},
    }
    assert _validate_baseline_or_report(base_v1) is True


def test_to_json_sanitizes_non_serializable(tmp_path: Path) -> None:
    rp = _mk_report()
    # Inject a non-serializable object; sanitizer should render it as string
    rp["meta"]["extra"] = Path("/tmp/unit").resolve()
    txt = to_json(rp)
    assert "/tmp" in txt or "unit" in txt
