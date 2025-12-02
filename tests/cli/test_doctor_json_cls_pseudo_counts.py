import json
from unittest.mock import Mock, patch

from typer.testing import CliRunner


def _mk_report_with_pseudo_accuracy(profile: str = "dev") -> dict:
    return {
        "meta": {"profile": profile},
        "metrics": {
            "primary_metric": {
                "kind": "accuracy",
                "final": 1.0,
                "display_ci": [1.0, 1.0],
                "counts_source": "pseudo_config",
                "estimated": True,
            }
        },
    }


def test_doctor_reports_d012_warning_for_dev_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")

    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(json.dumps(_mk_report_with_pseudo_accuracy("dev")))
    subject.write_text(json.dumps(_mk_report_with_pseudo_accuracy("dev")))

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
    assert "D012" in codes
    # Should be a warning in dev
    sev = {f["code"]: f["severity"] for f in payload["findings"]}
    assert sev.get("D012") == "warning"


def test_doctor_reports_d012_error_for_ci_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")

    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(json.dumps(_mk_report_with_pseudo_accuracy("ci")))
    subject.write_text(json.dumps(_mk_report_with_pseudo_accuracy("ci")))

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
                "--profile",
                "ci",
                "--baseline-report",
                str(baseline),
                "--subject-report",
                str(subject),
            ],
        )

    payload = json.loads(r.stdout)
    codes = {f["code"] for f in payload["findings"]}
    assert "D012" in codes
    sev = {f["code"]: f["severity"] for f in payload["findings"]}
    assert sev.get("D012") == "error"
    assert payload["resolution"]["exit_code"] == 1
