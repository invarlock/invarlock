from __future__ import annotations

import copy
from pathlib import Path

from invarlock.core.assurance_contract import (
    CANONICAL_GUARD_CHAIN,
    build_assurance_section,
)
from invarlock.reporting.verify_contract import VerifyOutcome
from tests.cli.verify._support_runtime_provenance import bind_runtime_policy_receipt
from tests.core._support_spectral_replay import _over_budget_report
from tests.reporting.validation._support_verify_assurance_guard_chain import (
    _report,
    _run_strict,
    _verified_runtime,
    _write_report,
)


def test_verify_assurance_strict_accepts_observed_replayed_spectral_finding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    observed = _over_budget_report()
    measurement_contract = copy.deepcopy(payload["spectral"]["measurement_contract"])
    measurement_contract_hash = payload["spectral"]["measurement_contract_hash"]
    payload["guards"][1] = observed["guards"][0]
    payload["spectral"] = observed["spectral"]
    payload["spectral"].update(
        {
            "measurement_contract": measurement_contract,
            "measurement_contract_hash": measurement_contract_hash,
            "measurement_contract_match": True,
        }
    )
    payload["validation"]["spectral_stable"] = False
    payload["resolved_policy"]["guard_authority"]["spectral"] = "observe"
    bind_runtime_policy_receipt(payload)
    payload["assurance"] = build_assurance_section(payload)
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.OK, "\n".join(
        item.message for item in result.diagnostics
    )


def test_verify_assurance_strict_accepts_observed_replayed_rmt_finding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    finding = {
        "family": "ffn",
        "edge_base": 1.0,
        "edge_cur": 1.02,
        "allowed": 1.01,
        "epsilon": 0.01,
        "delta": 0.02,
    }
    payload["rmt"].update(
        {
            "passed": False,
            "decision": "block",
            "status": "unstable",
            "stable": False,
            "epsilon_violations": [copy.deepcopy(finding)],
            "edge_risk_by_family": {"ffn": 1.02},
        }
    )
    payload["rmt"]["families"]["ffn"].update(
        {"edge_cur": 1.02, "ratio": 1.02, "delta": 0.02}
    )
    raw = payload["guards"][2]
    raw.update(
        {
            "passed": False,
            "decision": "block",
            "violations": [
                {
                    "type": "epsilon_band",
                    "severity": "error",
                    **finding,
                }
            ],
            "diagnostics": [
                {
                    "kind": "epsilon_band",
                    "severity": "error",
                    "message": "measured epsilon violation",
                }
            ],
        }
    )
    raw["metrics"].update(
        {
            "stable": False,
            "edge_risk_by_family": {"ffn": 1.02},
            "edge_risk_by_module": {"layer.0.mlp": 1.02},
            "epsilon_violations": [copy.deepcopy(finding)],
        }
    )
    payload["validation"]["rmt_stable"] = False
    payload["resolved_policy"]["guard_authority"]["rmt"] = "observe"
    bind_runtime_policy_receipt(payload)
    payload["assurance"] = build_assurance_section(payload)
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.OK, "\n".join(
        item.message for item in result.diagnostics
    )
