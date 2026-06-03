from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app
from tests.reporting._support_evidence_pack_paths import (
    _build_report_payload,
    _successful_verify_result,
    _write_json,
    _write_runtime_manifest,
)


def test_evidence_pack_verify_expected_fingerprint_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    final_verdict = tmp_path / "final_verdict.json"
    report = tmp_path / "evaluation.report.json"
    out_dir = tmp_path / "evidence_pack_signed"
    private_key = tmp_path / "evidence-pack-signing-key.pem"

    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(report, _build_report_payload())
    _write_runtime_manifest(report)
    monkeypatch.setattr(
        "invarlock.evidence_pack._run_verify_command",
        lambda reports, profile, report_assurance="report": _successful_verify_result(
            reports
        ),
        raising=False,
    )

    keygen = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "keygen",
            str(private_key),
            "--json",
        ],
    )
    assert keygen.exit_code == 0, keygen.output
    fingerprint = json.loads(keygen.stdout.strip())["signing_key_fingerprint"]

    build = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "build",
            str(out_dir),
            "--final-verdict",
            str(final_verdict),
            "--report",
            str(report),
            "--signing-key",
            str(private_key),
            "--json",
        ],
    )
    assert build.exit_code == 0, build.output

    pinned = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "verify",
            str(out_dir),
            "--expected-fingerprint",
            fingerprint,
            "--json",
        ],
    )
    assert pinned.exit_code == 0, pinned.output
    pinned_payload = json.loads(pinned.stdout.strip())
    assert pinned_payload["ok"] is True
    assert pinned_payload["authenticity"] == "pinned"

    mismatch = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "verify",
            str(out_dir),
            "--expected-fingerprint",
            "sha256:" + ("0" * 64),
            "--json",
        ],
    )
    assert mismatch.exit_code == 5, mismatch.output
    mismatch_payload = json.loads(mismatch.stdout.strip())
    assert mismatch_payload["ok"] is False
    assert mismatch_payload["authenticity"] == "mismatch"
    assert "signer mismatch" in mismatch_payload["errors"][0]
