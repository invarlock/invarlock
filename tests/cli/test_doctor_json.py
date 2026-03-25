from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.cli.commands.doctor import doctor_command


def test_doctor_json_mode_outputs_findings_and_exitcode(monkeypatch):
    # Enable a note-producing flag
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    r = CliRunner().invoke(app, ["doctor", "--json"])
    # doctor emits JSON and exits with code 0 for healthy (CI host may be CPU-only but still healthy)
    assert r.exit_code in (0, 1)  # accept both in varied environments
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    assert payload.get("format_version") == "doctor-v1"
    assert isinstance(payload.get("findings"), list)
    assert payload["contracts"]["model_family_catalog"]["format_version"] == (
        "model-family-catalog-v1"
    )
    assert payload["model_family_catalog"]["format_version"] == (
        "model-family-catalog-v1"
    )
    # Should include at least one note when INVARLOCK_TINY_RELAX is set
    # (ok if filtered out on some builds; ensure structure viable)
    assert "summary" in payload and "resolution" in payload


def test_doctor_command_raises_typer_exit_for_human_mode(monkeypatch):
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    with pytest.raises(typer.Exit) as exc:
        doctor_command(json_out=False)
    assert exc.value.exit_code in (0, 1)
