import json

from typer.testing import CliRunner


def test_doctor_json_includes_format_version(monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    from invarlock.cli.app import app

    res = CliRunner().invoke(app, ["doctor", "--json"])  # pure JSON payload
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
