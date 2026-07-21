"""Adversarial coverage for the public command surface."""

from __future__ import annotations

import importlib
import json
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import invarlock.evaluation_transaction as evaluation_transaction
import invarlock.evidence_reporting as evidence_reporting
import invarlock.evidence_verification as evidence_verification
from invarlock.cli.app import app
from invarlock.core import evaluation_request as evaluation_request_module

cli_module = importlib.import_module("invarlock.cli.app")
_RUNNER = CliRunner()


def _mock_loaded_import_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluation_request_module,
        "load_evaluation_request",
        lambda *_args, **_kwargs: SimpleNamespace(
            execution=SimpleNamespace(mode="import")
        ),
    )
    monkeypatch.setattr(
        evaluation_transaction,
        "preflight_evaluation_request",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )


def test_version_callback_emits_installed_and_source_fallback_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(cli_module, "console", SimpleNamespace(print=rendered.append))
    monkeypatch.setattr(cli_module, "version", lambda _name: "1.2.3")

    with pytest.raises(Exception) as installed_exit:
        cli_module._version_callback(True)
    assert installed_exit.value.__class__.__name__ == "Exit"
    assert rendered == ["InvarLock 1.2.3"]

    rendered.clear()
    monkeypatch.setattr(
        cli_module,
        "version",
        lambda _name: (_ for _ in ()).throw(PackageNotFoundError("invarlock")),
    )
    cli_module._emit_version()
    assert rendered and rendered[0].startswith("InvarLock ")
    cli_module._version_callback(False)


def test_evaluate_renders_success_in_human_and_json_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = tmp_path / "request.yaml"
    request.write_text("format_version: fixture\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    result = evaluation_transaction.EvaluationTransactionResult(
        evidence_path=evidence,
        comparison_id="comparison-123",
        pack_manifest_digest="sha256:" + ("a" * 64),
    )
    monkeypatch.setattr(
        evaluation_transaction,
        "evaluate_request_file",
        lambda *_args, **_kwargs: result,
    )
    _mock_loaded_import_request(monkeypatch)

    human = _RUNNER.invoke(app, ["evaluate", str(request)])
    machine = _RUNNER.invoke(app, ["evaluate", str(request), "--json"])

    assert human.exit_code == 0
    assert "PASS Evidence pack published" in human.stdout
    assert str(evidence) in human.stdout.replace("\n", "")
    assert machine.exit_code == 0
    assert json.loads(machine.stdout) == json.loads(result.as_json())


@pytest.mark.parametrize("json_out", [False, True])
def test_evaluate_preserves_transaction_failure_code_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    json_out: bool,
) -> None:
    request = tmp_path / "request.yaml"
    request.write_text("format_version: fixture\n", encoding="utf-8")

    def fail(*_args: object, **_kwargs: object) -> object:
        raise evaluation_transaction.EvaluationTransactionError(
            "runtime digest is not independently bound",
            exit_code=7,
        )

    monkeypatch.setattr(evaluation_transaction, "evaluate_request_file", fail)
    _mock_loaded_import_request(monkeypatch)
    arguments = ["evaluate", str(request)]
    if json_out:
        arguments.append("--json")

    result = _RUNNER.invoke(app, arguments)

    assert result.exit_code == 7
    if json_out:
        assert json.loads(result.stdout)["errors"] == [
            "runtime digest is not independently bound"
        ]
    else:
        assert "FAIL runtime digest is not independently bound" in result.stdout


def test_verify_renders_success_in_human_and_json_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    verified = evidence_verification.EvidenceVerification(
        evidence_path=evidence,
        payload={
            "format_version": "invarlock/evidence-verification-v1",
            "ok": True,
            "comparison_id": "comparison-123",
            "signer_fingerprint": "sha256:" + "a" * 64,
        },
    )
    monkeypatch.setattr(
        evidence_verification,
        "verify_evidence",
        lambda *_args, **_kwargs: verified,
    )

    human = _RUNNER.invoke(app, ["verify", str(evidence)])
    machine = _RUNNER.invoke(app, ["verify", str(evidence), "--json"])

    assert human.exit_code == 0
    assert "PASS Evidence verified" in human.stdout
    assert "Comparison: comparison-123" in human.stdout
    assert machine.exit_code == 0
    assert json.loads(machine.stdout)["ok"] is True


@pytest.mark.parametrize("json_out", [False, True])
def test_verify_preserves_signed_failure_receipt_and_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    json_out: bool,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    receipt = tmp_path / "failed-verification-receipt.json"
    payload = {
        "format_version": "invarlock/evidence-verification-v1",
        "ok": False,
        "errors": ["subject runtime digest mismatch"],
        "warnings": [],
        "signed_receipt": str(receipt),
    }

    def fail(*_args: object, **_kwargs: object) -> object:
        raise evidence_verification.EvidenceVerificationError(
            "subject runtime digest mismatch",
            exit_code=9,
            payload=payload,
        )

    monkeypatch.setattr(evidence_verification, "verify_evidence", fail)
    arguments = ["verify", str(evidence)]
    if json_out:
        arguments.append("--json")

    result = _RUNNER.invoke(app, arguments)

    assert result.exit_code == 9
    if json_out:
        assert json.loads(result.stdout) == payload
    else:
        assert "FAIL subject runtime digest mismatch" in result.stdout
        assert f"Receipt {receipt}" in result.stdout.replace("\n", "")


def test_verify_human_failure_without_signed_receipt_does_not_invent_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()

    def fail(*_args: object, **_kwargs: object) -> object:
        raise evidence_verification.EvidenceVerificationError(
            "policy input is required",
            exit_code=2,
        )

    monkeypatch.setattr(evidence_verification, "verify_evidence", fail)

    result = _RUNNER.invoke(app, ["verify", str(evidence)])

    assert result.exit_code == 2
    assert "FAIL policy input is required" in result.stdout
    assert "Receipt" not in result.stdout


def test_report_renders_text_and_html_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    html = tmp_path / "report.html"
    monkeypatch.setattr(
        evidence_reporting,
        "render_evidence",
        lambda *_args, **_kwargs: evidence_reporting.EvidenceReport(
            text="# InvarLock comparison report",
            html_path=html,
            evidence_signer="sha256:" + "a" * 64,
            pack_manifest_digest="sha256:" + "b" * 64,
        ),
    )

    result = _RUNNER.invoke(
        app,
        ["report", str(evidence), "--html", str(html), "--explain"],
    )

    assert result.exit_code == 0
    assert "# InvarLock comparison report" in result.stdout
    assert f"HTML {html}" in result.stdout.replace("\n", "")


def test_report_json_binds_the_rendered_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    html = tmp_path / "report.html"
    digest = "sha256:" + "b" * 64
    monkeypatch.setattr(
        evidence_reporting,
        "render_evidence",
        lambda *_args, **_kwargs: evidence_reporting.EvidenceReport(
            text="# InvarLock comparison report",
            html_path=html,
            evidence_signer="sha256:" + "a" * 64,
            pack_manifest_digest=digest,
        ),
    )

    result = _RUNNER.invoke(
        app,
        ["report", str(evidence), "--html", str(html), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "format_version": "invarlock/evidence-report-v1",
        "html": str(html),
        "ok": True,
        "pack_manifest_digest": digest,
    }


def test_report_preserves_renderer_failure_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()

    def fail(*_args: object, **_kwargs: object) -> object:
        raise evidence_reporting.EvidenceReportError(
            "canonical report is not bound by the pack",
            exit_code=8,
        )

    monkeypatch.setattr(evidence_reporting, "render_evidence", fail)

    result = _RUNNER.invoke(app, ["report", str(evidence)])

    assert result.exit_code == 8
    assert "FAIL canonical report is not bound by the pack" in result.stdout
