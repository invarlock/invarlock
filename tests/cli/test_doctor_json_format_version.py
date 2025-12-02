import json

from typer.testing import CliRunner


def test_doctor_json_includes_format_version(monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    from invarlock.cli.app import app

    res = CliRunner().invoke(app, ["doctor", "--json"])  # pure JSON payload
    assert res.exit_code in (0, 1)
    payload = json.loads(res.stdout)
    assert payload.get("format_version") == "doctor-v1"
