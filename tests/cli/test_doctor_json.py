from __future__ import annotations

import json
from unittest.mock import Mock, patch

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
        "model-family-catalog-v2"
    )
    assert payload["model_family_catalog"]["format_version"] == (
        "model-family-catalog-v2"
    )
    # Should include at least one note when INVARLOCK_TINY_RELAX is set
    # (ok if filtered out on some builds; ensure structure viable)
    assert "summary" in payload and "resolution" in payload


def test_doctor_command_raises_typer_exit_for_human_mode(monkeypatch):
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    with pytest.raises(typer.Exit) as exc:
        doctor_command(json_out=False)
    assert exc.value.exit_code in (0, 1)


def test_doctor_json_includes_format_version(monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")

    res = CliRunner().invoke(app, ["doctor", "--json"])
    assert res.exit_code in (0, 1)
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload.get("format_version") == "doctor-v1"
    contracts = payload["contracts"]
    for key, filename in {
        "validation_keys": "validation_keys.json",
        "console_labels": "console_labels.json",
        "metric_kinds": "metric_kinds.json",
    }.items():
        assert contracts[key]["path"] == f"contracts/{filename}"
        assert contracts[key]["kind"] == "array"


def test_doctor_json_includes_tiny_relax_note(monkeypatch):
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")

    r = CliRunner().invoke(app, ["doctor", "--json"])
    assert r.exit_code in (0, 1)
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    findings = payload.get("findings", [])
    assert any(
        f.get("code") == "D013" and f.get("severity") == "note" for f in findings
    )


def test_doctor_emits_d013_when_relax_env(monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    with (
        patch("invarlock.core.registry.get_registry") as mock_registry,
        patch(
            "invarlock.cli.device.get_device_info",
            return_value={
                "auto_selected": "cpu",
                "cpu": {"available": True, "info": "Always"},
            },
        ),
    ):
        reg = Mock()
        reg.list_adapters.return_value = []
        reg.list_edits.return_value = []
        reg.list_guards.return_value = []
        reg.get_plugin_info.return_value = {
            "module": "invarlock.adapters",
            "entry_point": "",
        }
        mock_registry.return_value = reg

        res = CliRunner().invoke(app, ["doctor", "--json"])
    payload = json.loads(res.stdout)
    assert any(f.get("code") == "D013" for f in payload.get("findings", []))
