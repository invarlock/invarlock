import json
from unittest.mock import Mock, patch

from typer.testing import CliRunner


def _mk_report(tokenizer=None, masking=None, split=None, pm_kind=None):
    prov = {}
    if split is not None:
        prov["dataset_split"] = split
    pd = {}
    if tokenizer is not None:
        pd["tokenizer_sha256"] = tokenizer
    if masking is not None:
        pd["masking_sha256"] = masking
    if pd:
        prov["provider_digest"] = pd
    metrics = {}
    if pm_kind is not None:
        metrics = {"primary_metric": {"kind": pm_kind}}
    return {"provenance": prov, "metrics": metrics}


def test_doctor_json_reports_tokenizer_digest_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")

    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(json.dumps(_mk_report(tokenizer="tokA", split="validation")))
    subject.write_text(json.dumps(_mk_report(tokenizer="tokB", split="validation")))

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

    assert r.exit_code == 0  # warning only
    payload = json.loads(r.stdout)
    codes = {f["code"] for f in payload["findings"]}
    assert "D009" in codes


def test_doctor_json_reports_mask_digest_missing_for_mlm_like(tmp_path, monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")

    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(
        json.dumps(
            _mk_report(
                tokenizer="tokA", masking=None, split="validation", pm_kind="ppl_mlm"
            )
        )
    )
    subject.write_text(
        json.dumps(
            _mk_report(
                tokenizer="tokA", masking=None, split="validation", pm_kind="ppl_mlm"
            )
        )
    )

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

    assert r.exit_code == 0  # warning only
    payload = json.loads(r.stdout)
    codes = {f["code"] for f in payload["findings"]}
    assert "D010" in codes


def test_doctor_json_d010_not_emitted_when_not_mlm(tmp_path, monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")

    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(
        json.dumps(
            _mk_report(
                tokenizer="tokA", masking=None, split="validation", pm_kind="accuracy"
            )
        )
    )
    subject.write_text(
        json.dumps(
            _mk_report(
                tokenizer="tokA", masking=None, split="validation", pm_kind="accuracy"
            )
        )
    )

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
    assert "D010" not in codes


def test_doctor_json_reports_split_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")

    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(json.dumps(_mk_report(tokenizer="tokA", split="validation")))
    subject.write_text(json.dumps(_mk_report(tokenizer="tokA", split="test")))

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

    assert r.exit_code == 0  # warning only
    payload = json.loads(r.stdout)
    codes = {f["code"] for f in payload["findings"]}
    assert "D011" in codes


def test_d011_escalates_with_strict(tmp_path, monkeypatch):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")

    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text(json.dumps(_mk_report(tokenizer="tokA", split="validation")))
    subject.write_text(json.dumps(_mk_report(tokenizer="tokA", split="test")))

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
                "--strict",
            ],
        )

    payload = json.loads(r.stdout)
    findings = payload["findings"]
    sev = {f["code"]: f["severity"] for f in findings}
    assert sev.get("D011") == "error"
    assert payload["resolution"]["exit_code"] == 1
