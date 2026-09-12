from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import invarlock.evaluator_qualification_cli as qualification_cli
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


def test_cli_prints_verdict_authority_summary(tmp_path: Path) -> None:
    profile, schedule, export, raw = qualification_fixture(tmp_path)

    result = RUNNER.invoke(
        app,
        ["qualify", str(profile), str(schedule), str(export), str(raw)],
    )

    assert result.exit_code == 0, result.output
    assert "qualified for runtime import" in " ".join(result.stdout.split())


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


@pytest.mark.parametrize("json_out", [False, True])
def test_cli_escapes_controls_in_dynamic_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_out: bool
) -> None:
    paths = []
    for name in ("profile", "schedule", "export", "raw"):
        path = tmp_path / name
        path.write_text("fixture", encoding="utf-8")
        paths.append(path)
    payload = {
        "authority": "observation_only",
        "profile_id": "profile\x9b2J\nFAKE_PASS\u202e[bold]literal[/bold]",
        "reason_codes": ["reason\x9b2J\t\u202e"],
    }
    monkeypatch.setattr(
        qualification_cli,
        "qualify_evaluator_export",
        lambda **_kwargs: SimpleNamespace(
            authority="observation_only",
            profile_id=payload["profile_id"],
            reason_codes=tuple(payload["reason_codes"]),
            record_count=0,
            as_json=lambda: json.dumps(payload, ensure_ascii=False),
        ),
    )

    arguments = ["qualify", *(str(path) for path in paths)]
    if json_out:
        arguments.append("--json")
    result = RUNNER.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert "\x9b" not in result.stdout
    assert "\u202e" not in result.stdout
    if json_out:
        assert json.loads(result.stdout) == payload
    else:
        assert (
            "profile\\u009b2J\\u000aFAKE_PASS\\u202e[bold]literal[/bold]"
            in result.stdout
        )
        assert "reason\\u009b2J\\u0009\\u202e" in result.stdout
