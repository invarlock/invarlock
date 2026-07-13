from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app
from tests.reporting._support_evidence_pack_paths import (
    _build_pack,
    _build_report_payload,
    _sign_pack,
    _successful_verify_result,
)


def test_evidence_pack_verify_expected_fingerprint_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "evidence_pack_signed"

    _build_pack(
        out_dir,
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
        report_payload=_build_report_payload(),
    )
    fingerprint = _sign_pack(out_dir, tmp_path)
    monkeypatch.setattr(
        "invarlock.evidence_pack._run_verify_command",
        lambda reports, profile, report_assurance="report": _successful_verify_result(
            reports
        ),
        raising=False,
    )

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
