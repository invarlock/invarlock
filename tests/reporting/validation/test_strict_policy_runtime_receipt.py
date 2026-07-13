from __future__ import annotations

from invarlock.core.assurance_contract import CANONICAL_GUARD_CHAIN
from invarlock.reporting.verify_contract import VerifyOutcome
from tests.reporting.validation._support_verify_assurance_guard_chain import (
    _report,
    _run_strict,
    _verified_runtime,
    _write_report,
)


def test_strict_verify_rejects_missing_runtime_policy_receipt(
    tmp_path, monkeypatch
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload.pop("policy_resolution")
    path = tmp_path / "evaluation.report.json"
    _write_report(path, payload)

    result = _run_strict(path)

    assert result.outcome is VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "requires a runtime policy receipt" in diagnostics


def test_strict_verify_rejects_runtime_policy_receipt_tamper(
    tmp_path, monkeypatch
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["resolved_policy"]["spectral"]["max_caps"] = 999
    path = tmp_path / "evaluation.report.json"
    _write_report(path, payload)

    result = _run_strict(path)

    assert result.outcome is VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "receipt digest does not match" in diagnostics
