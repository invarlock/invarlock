import json
from unittest.mock import Mock, patch

from typer.testing import CliRunner


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

        from invarlock.cli.app import app

        res = CliRunner().invoke(app, ["doctor", "--json"])
    payload = json.loads(res.stdout)
    assert any(f.get("code") == "D013" for f in payload.get("findings", []))
