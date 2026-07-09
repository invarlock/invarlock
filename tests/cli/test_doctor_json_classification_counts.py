from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner


def _mk_accuracy_report(
    *,
    profile: str = "dev",
    counts_source: str,
    estimated: bool,
    final: float = 1.0,
) -> dict:
    return {
        "meta": {"profile": profile},
        "metrics": {
            "primary_metric": {
                "kind": "accuracy",
                "final": final,
                "display_ci": [final, final],
                "counts_source": counts_source,
                "estimated": estimated,
            }
        },
    }


def _write_reports(tmp_path, report: dict) -> tuple[str, str]:
    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(json.dumps(report), encoding="utf-8")
    subject.write_text(json.dumps(report), encoding="utf-8")
    return str(baseline), str(subject)


def _invoke_doctor_json(monkeypatch, args: list[str]):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
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

        return CliRunner().invoke(app, ["doctor", "--json", *args])


def test_doctor_measured_cls_no_d012(tmp_path, monkeypatch):
    baseline, subject = _write_reports(
        tmp_path,
        _mk_accuracy_report(counts_source="measured", estimated=False, final=0.75),
    )

    result = _invoke_doctor_json(
        monkeypatch,
        ["--baseline-report", baseline, "--subject-report", subject],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    codes = {f["code"] for f in payload["findings"]}
    assert "D012" not in codes


@pytest.mark.parametrize(
    ("profile", "args", "expected_severity", "expected_exit_code"),
    (
        ("dev", [], "warning", 0),
        ("ci", ["--profile", "ci"], "error", 1),
    ),
)
def test_doctor_reports_d012_for_pseudo_classification_counts(
    tmp_path,
    monkeypatch,
    profile: str,
    args: list[str],
    expected_severity: str,
    expected_exit_code: int,
):
    baseline, subject = _write_reports(
        tmp_path,
        _mk_accuracy_report(
            profile=profile,
            counts_source="pseudo_config",
            estimated=True,
        ),
    )

    result = _invoke_doctor_json(
        monkeypatch,
        [*args, "--baseline-report", baseline, "--subject-report", subject],
    )

    payload = json.loads(result.stdout)
    codes = {f["code"] for f in payload["findings"]}
    severity_by_code = {f["code"]: f["severity"] for f in payload["findings"]}
    assert "D012" in codes
    assert severity_by_code.get("D012") == expected_severity
    assert payload["resolution"]["exit_code"] == expected_exit_code
