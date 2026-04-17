from __future__ import annotations

import json
from pathlib import Path

from invarlock.cli.commands import verify as verify_cmd_mod
from invarlock.cli.commands.verify import verify_command
from invarlock.reporting.verify_contract import (
    VerifyDiagnostic,
    VerifyExecutionResult,
    VerifyOutcome,
)


def _write_cert(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_cert(ratio: float = 2.0) -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r1",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "ratio_vs_baseline": ratio,
            "display_ci": [0.98, 1.02],
        },
        "dataset": {
            "provider": "unit",
            "seq_len": 8,
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                    "paired_windows": 1,
                },
            },
        },
        "baseline_ref": {"primary_metric": {"final": 5.0}},
    }


def test_verify_human_ok_line(tmp_path: Path, capsys) -> None:
    c = _write_cert(tmp_path / "ok.json", _valid_cert())
    verify_command([c], baseline=None, tolerance=1e-9, profile="dev", json_out=False)
    out = capsys.readouterr().out
    assert "VERIFY OK" in out


def test_verify_human_fail_output(tmp_path: Path, capsys) -> None:
    bad = _valid_cert(ratio=1.0)
    c = _write_cert(tmp_path / "bad.json", bad)
    # For human output, the command prints FAIL lines; it may not raise SystemExit
    try:
        verify_command(
            [c], baseline=None, tolerance=1e-9, profile="dev", json_out=False
        )
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "FAIL" in out or "policy_fail" in out or "Primary metric" in out


def test_verify_human_ok_line_missing_optional_fields(tmp_path: Path, capsys) -> None:
    cert = _valid_cert()
    # Remove optional pieces so human line covers alternate branches
    del cert["primary_metric"]["display_ci"]
    c = _write_cert(tmp_path / "ok2.json", cert)
    verify_command([c], baseline=None, tolerance=1e-9, profile="dev", json_out=False)
    out = capsys.readouterr().out
    assert "VERIFY OK" in out


def test_verify_human_ci_policy_fail(tmp_path: Path, capsys) -> None:
    # Missing provider digest in CI should raise InvarlockError and print human error
    cert = _valid_cert()
    # Provide minimal provenance without provider_digest
    cert["provenance"] = {}
    c = _write_cert(tmp_path / "ci_bad.json", cert)
    try:
        verify_command([c], baseline=None, tolerance=1e-9, profile="ci", json_out=False)
    except SystemExit as ex:
        # Exit code 3 in CI for InvarlockError
        assert ex.code == 3
    out = capsys.readouterr().out
    # Either prints explicit failure message or surfaces an error summary
    assert (
        "Verification failed" in out
        or "policy_fail" in out
        or "PROVIDER-DIGEST-MISSING" in out
    )


def test_verify_human_error_diagnostic_line(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(
        verify_cmd_mod,
        "run_verify_reports",
        lambda *args, **kwargs: VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True},
            diagnostics=(VerifyDiagnostic(level="error", message="runtime mismatch"),),
        ),
        raising=True,
    )

    verify_command([tmp_path / "ok.json"], baseline=None, profile="dev", json_out=False)

    out = capsys.readouterr().out
    assert "runtime mismatch" in out
