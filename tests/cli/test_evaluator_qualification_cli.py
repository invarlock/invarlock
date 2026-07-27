from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.evaluator_qualification_cli import app
from tests.core.test_evaluator_qualification import qualification_fixture

RUNNER = CliRunner()


def test_cli_qualifies_custom_export_and_writes_result(tmp_path: Path) -> None:
    profile, schedule, export, raw = qualification_fixture(tmp_path)
    output = tmp_path / "qualification.json"

    result = RUNNER.invoke(
        app,
        [
            "qualify",
            str(profile),
            str(schedule),
            str(export),
            str(raw),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["outcome"] == "qualified_for_import"
    assert payload["authority"] == "verdict_authority"
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_cli_can_require_verdict_authority(tmp_path: Path) -> None:
    profile, schedule, export, raw = qualification_fixture(
        tmp_path,
        mode="observation_only",
    )

    result = RUNNER.invoke(
        app,
        [
            "qualify",
            str(profile),
            str(schedule),
            str(export),
            str(raw),
            "--require-verdict-authority",
        ],
    )

    assert result.exit_code == 3
    assert "observation-only" in result.stdout


def test_cli_reports_digest_failure_without_traceback(tmp_path: Path) -> None:
    profile, schedule, export, raw = qualification_fixture(tmp_path)
    raw.write_bytes(b"tampered\n")

    result = RUNNER.invoke(
        app,
        ["qualify", str(profile), str(schedule), str(export), str(raw)],
    )

    assert result.exit_code == 2
    assert "raw upstream output digest does not match" in result.stdout
    assert "Traceback" not in result.stdout

