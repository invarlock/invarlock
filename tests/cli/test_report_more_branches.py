from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.models import OptionInfo

import invarlock.cli.commands.report as report_mod
import invarlock.reporting.certificate as cert_mod
import invarlock.reporting.report as report_lib


def _make_primary_report():
    return {
        "meta": {"model_id": "subject"},
        "edit": {"name": "quant_rtn"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "final": 10.0,
                "display_ci": [9.5, 10.5],
                "ratio_vs_baseline": 1.02,
            }
        },
    }


def test_generate_reports_coerces_optioninfo_and_all_formats(monkeypatch):
    monkeypatch.setattr(
        report_mod, "_load_run_report", lambda path: _make_primary_report()
    )
    saved = {}

    def fake_save(primary, out_dir, *, formats, **kwargs):
        saved["formats"] = formats
        return {fmt: f"{fmt}.json" for fmt in formats}

    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )
    monkeypatch.setattr(report_lib, "save_report", fake_save, raising=False)

    run_opt = OptionInfo()
    run_opt.default = "run.json"
    format_opt = OptionInfo()
    format_opt.default = "all"
    report_mod._generate_reports(
        run=run_opt,
        format=format_opt,
        compare=None,
        baseline=None,
        output=None,
    )
    assert saved["formats"] == ["json", "markdown", "html"]


def test_generate_reports_certificate_validation_block(monkeypatch):
    primary = _make_primary_report()
    baseline = _make_primary_report()

    def fake_load(path):
        return baseline if "baseline" in path else primary

    monkeypatch.setattr(report_mod, "_load_run_report", fake_load, raising=False)

    monkeypatch.setattr(
        report_lib,
        "save_report",
        lambda *_, **__: {"cert": "evaluation.cert.json"},
        raising=False,
    )
    monkeypatch.setattr(
        cert_mod,
        "make_certificate",
        lambda *_, **__: {"validation": {"overall": True}},
        raising=False,
    )
    monkeypatch.setattr(
        cert_mod, "validate_certificate", lambda cert: True, raising=False
    )

    block = {
        "overall_pass": True,
        "rows": [{"label": "primary_metric", "status": "PASS"}],
    }

    def fake_console_block(cert):
        return block

    monkeypatch.setattr(
        "invarlock.reporting.render.compute_console_validation_block",
        fake_console_block,
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    report_mod.report_command(
        run="run.json",
        format="cert",
        compare=None,
        baseline="baseline.json",
        output="out",
    )


def test_generate_reports_certificate_validation_error(monkeypatch):
    monkeypatch.setattr(
        report_mod, "_load_run_report", lambda path: _make_primary_report()
    )
    monkeypatch.setattr(
        report_lib,
        "save_report",
        lambda *_, **__: {"cert": "evaluation.cert.json"},
        raising=False,
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("bad cert")

    monkeypatch.setattr(cert_mod, "make_certificate", _boom, raising=False)
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_command(
            run="run.json",
            format="cert",
            compare=None,
            baseline="baseline.json",
            output=None,
        )
    assert exc.value.exit_code == 1


def test_report_validate_success(monkeypatch, tmp_path):
    cert = tmp_path / "cert.json"
    cert.write_text(json.dumps({"ok": True}), encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )
    monkeypatch.setattr(
        cert_mod,
        "validate_certificate",
        lambda payload: True,
        raising=False,
    )
    report_mod.report_validate(report=str(cert))


def test_report_validate_schema_failure(monkeypatch, tmp_path):
    cert = tmp_path / "cert.json"
    cert.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )
    monkeypatch.setattr(
        cert_mod,
        "validate_certificate",
        lambda payload: False,
        raising=False,
    )
    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(cert))
    assert exc.value.exit_code == 2


def test_report_validate_value_error(monkeypatch, tmp_path):
    cert = tmp_path / "cert.json"
    cert.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    def _raise_val(payload):
        raise ValueError("bad schema")

    monkeypatch.setattr(cert_mod, "validate_certificate", _raise_val, raising=False)
    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(cert))
    assert exc.value.exit_code == 2


def test_report_validate_generic_error(monkeypatch, tmp_path):
    cert = tmp_path / "cert.json"
    cert.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    def _raise(payload):
        raise RuntimeError("boom")

    monkeypatch.setattr(cert_mod, "validate_certificate", _raise, raising=False)
    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(cert))
    assert exc.value.exit_code == 1


def test_report_validate_read_failure(monkeypatch, tmp_path):
    cert = tmp_path / "cert.json"
    cert.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, **kwargs: (_ for _ in ()).throw(OSError("io fail")),
        raising=False,
    )
    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(cert))
    assert exc.value.exit_code == 1
