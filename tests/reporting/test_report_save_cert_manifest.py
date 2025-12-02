from __future__ import annotations

from pathlib import Path

import pytest

from invarlock.reporting.report import save_report
from invarlock.reporting.report_types import RunReport, create_empty_report


def _mk_simple_report() -> RunReport:
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
    # Include a tiny guard context so evidence helper runs
    r["guards"] = [
        {
            "name": "variance",
            "policy": {"deadband": 0.1},
            "metrics": {},
            "actions": [],
            "violations": [],
        }
    ]
    return r


def test_save_report_cert_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rp = _mk_simple_report()
    base = _mk_simple_report()
    # Gate evidence emission on env
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")
    out = save_report(
        rp,
        tmp_path,
        formats=["cert"],
        compare=None,
        baseline=base,
        filename_prefix="unit",
    )
    # Ensure expected files are produced
    assert (tmp_path / "evaluation.cert.json").exists()
    assert out.get("cert") == (tmp_path / "evaluation.cert.json")
    assert (tmp_path / "unit_certificate.md").exists()
    assert (tmp_path / "manifest.json").exists()
