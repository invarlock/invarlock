import json
from unittest.mock import Mock, patch

from typer.testing import CliRunner


def _mk_measured_report() -> dict:
    return {
        "meta": {"profile": "dev"},
        "metrics": {
            "primary_metric": {
                "kind": "accuracy",
                "final": 0.75,
                "display_ci": [0.75, 0.75],
                "counts_source": "measured",
                "estimated": False,
            }
        },
    }


def test_doctor_measured_cls_no_d012(tmp_path, monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")

    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(json.dumps(_mk_measured_report()))
    subject.write_text(json.dumps(_mk_measured_report()))

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

        r = CliRunner().invoke(
            app,
            [
                "doctor",
                "--json",
                "--baseline-report",
                str(baseline),
                "--subject-report",
                str(subject),
            ],
        )

    assert r.exit_code == 0
    payload = json.loads(r.stdout)
    codes = {f["code"] for f in payload["findings"]}
    assert "D012" not in codes
