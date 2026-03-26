from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.models import OptionInfo

import invarlock.cli.commands.report as report_mod
import invarlock.reporting.report_builder as cert_mod


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
    monkeypatch.setattr(report_mod, "_save_report", fake_save, raising=False)

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


def test_generate_reports_evaluation_report_validation_block(monkeypatch):
    primary = _make_primary_report()
    baseline = _make_primary_report()

    def fake_load(path):
        return baseline if "baseline" in path else primary

    monkeypatch.setattr(report_mod, "_load_run_report", fake_load, raising=False)

    monkeypatch.setattr(
        report_mod,
        "_save_report",
        lambda *_, **__: {"report": "evaluation.report.json"},
        raising=False,
    )
    monkeypatch.setattr(
        cert_mod,
        "make_report",
        lambda *_, **__: {
            "validation": {"overall": True},
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 45.657,
                "final": 47.082,
                "ratio_vs_baseline": 1.0,
                "display_ci": [0.9981, 1.0019],
            },
        },
        raising=False,
    )
    monkeypatch.setattr(cert_mod, "validate_report", lambda cert: True, raising=False)

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
    captured: list[str] = []

    class _CaptureConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(report_mod, "console", _CaptureConsole())

    report_mod.report_command(
        run="run.json",
        format="report",
        compare=None,
        baseline="baseline.json",
        output="out",
    )
    out = "\n".join(captured)
    assert "PRIMARY METRIC" in out
    assert "CI (95%)" in out
    assert "[0.998, 1.002]" in out


def test_generate_reports_summary_includes_total_time_suffix(monkeypatch):
    primary = _make_primary_report()
    baseline = _make_primary_report()

    def fake_load(path):
        return baseline if "baseline" in path else primary

    monkeypatch.setattr(report_mod, "_load_run_report", fake_load, raising=False)
    monkeypatch.setattr(
        report_mod,
        "_save_report",
        lambda *_, **__: {"report": "evaluation.report.json"},
        raising=False,
    )
    monkeypatch.setattr(
        cert_mod,
        "make_report",
        lambda *_, **__: {"validation": {"overall": True}, "primary_metric": {}},
        raising=False,
    )
    monkeypatch.setattr(cert_mod, "validate_report", lambda cert: True, raising=False)
    monkeypatch.setattr(
        "invarlock.reporting.render.compute_console_validation_block",
        lambda cert: {"overall_pass": True, "rows": []},
    )
    monkeypatch.setattr(report_mod, "perf_counter", lambda: 126.0)

    captured: list[str] = []

    class _CaptureConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(report_mod, "console", _CaptureConsole())

    report_mod.report_command(
        run="run.json",
        format="report",
        compare=None,
        baseline="baseline.json",
        output="out",
        summary_baseline_seconds=1.0,
        summary_subject_seconds=2.0,
        summary_report_start=120.0,
    )
    out = "\n".join(captured)
    assert "EVALUATION REPORT SUMMARY" in out
    assert "[9.00s]" in out


def test_generate_reports_evaluation_report_validation_error(monkeypatch):
    monkeypatch.setattr(
        report_mod, "_load_run_report", lambda path: _make_primary_report()
    )
    monkeypatch.setattr(
        report_mod,
        "_save_report",
        lambda *_, **__: {"report": "evaluation.report.json"},
        raising=False,
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("bad report")

    monkeypatch.setattr(cert_mod, "make_report", _boom, raising=False)
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_command(
            run="run.json",
            format="report",
            compare=None,
            baseline="baseline.json",
            output=None,
        )
    assert exc.value.exit_code == 1


def test_generate_reports_rejects_noncanonical_run_directory(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "my_report.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_command(
            run=str(report_dir),
            format="json",
            compare=None,
            baseline=None,
            output=None,
        )

    assert exc.value.exit_code == 2


def test_report_validate_success(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"ok": True}), encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )
    monkeypatch.setattr(
        cert_mod,
        "validate_report",
        lambda payload: True,
        raising=False,
    )
    report_mod.report_validate(report=str(report))


def test_report_validate_accepts_canonical_directory(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "evaluation.report.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )
    monkeypatch.setattr(
        cert_mod,
        "validate_report",
        lambda payload: True,
        raising=False,
    )

    report_mod.report_validate(report=str(report_dir))


def test_report_validate_rejects_noncanonical_directory(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "my_report.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(report_dir))

    assert exc.value.exit_code == 2


def test_report_validate_schema_failure(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )
    monkeypatch.setattr(
        cert_mod,
        "validate_report",
        lambda payload: False,
        raising=False,
    )
    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(report))
    assert exc.value.exit_code == 2


def test_report_validate_value_error(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    def _raise_val(payload):
        raise ValueError("bad schema")

    monkeypatch.setattr(cert_mod, "validate_report", _raise_val, raising=False)
    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(report))
    assert exc.value.exit_code == 2


def test_report_validate_generic_error(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    def _raise(payload):
        raise RuntimeError("boom")

    monkeypatch.setattr(cert_mod, "validate_report", _raise, raising=False)
    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(report))
    assert exc.value.exit_code == 1


def test_report_validate_read_failure(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    original_open = Path.open

    def _bad_open(self, *args, **kwargs):
        if self == report:
            raise OSError("io fail")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(
        Path,
        "open",
        _bad_open,
        raising=False,
    )
    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(report))
    assert exc.value.exit_code == 2


def test_report_verify_command_resolves_canonical_directories(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    report_json = report_dir / "evaluation.report.json"
    report_json.write_text("{}", encoding="utf-8")
    baseline_dir = tmp_path / "baseline-dir"
    baseline_dir.mkdir()
    baseline_json = baseline_dir / "report.json"
    baseline_json.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "invarlock.cli.commands.verify.verify_command",
        lambda **kwargs: captured.update(kwargs),
        raising=False,
    )

    report_mod.report_verify_command(
        reports=[str(report_dir)],
        baseline=str(baseline_dir),
    )

    assert captured["reports"] == [report_json.resolve()]
    assert captured["baseline"] == baseline_json.resolve()


def test_report_verify_command_rejects_noncanonical_report_directory(
    monkeypatch, tmp_path
):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "my_report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_verify_command(reports=[str(report_dir)])

    assert exc.value.exit_code == 2


def test_report_verify_command_rejects_noncanonical_baseline_directory(
    monkeypatch, tmp_path
):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    baseline_dir = tmp_path / "baseline-dir"
    baseline_dir.mkdir()
    (baseline_dir / "subject_report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_verify_command(
            reports=[str(report_dir)],
            baseline=str(baseline_dir),
        )

    assert exc.value.exit_code == 2


def test_report_explain_resolves_canonical_directories(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    report_json = report_dir / "evaluation.report.json"
    report_json.write_text("{}", encoding="utf-8")
    baseline_dir = tmp_path / "baseline-dir"
    baseline_dir.mkdir()
    baseline_json = baseline_dir / "report.json"
    baseline_json.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "invarlock.cli.commands.explain_gates.explain_gates_command",
        lambda **kwargs: captured.update(kwargs),
        raising=False,
    )

    report_mod.report_explain(report=str(report_dir), baseline=str(baseline_dir))

    assert captured["report"] == str(report_json.resolve())
    assert captured["baseline"] == str(baseline_json.resolve())


@pytest.mark.parametrize("invalid_slot", ["report", "baseline"])
def test_report_explain_rejects_noncanonical_directory_inputs(
    monkeypatch, tmp_path, invalid_slot
):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    baseline_dir = tmp_path / "baseline-dir"
    baseline_dir.mkdir()
    (baseline_dir / "report.json").write_text("{}", encoding="utf-8")
    invalid_dir = report_dir if invalid_slot == "report" else baseline_dir
    canonical_file = (
        "subject_report.json" if invalid_slot == "report" else "my_report.json"
    )
    for candidate in invalid_dir.iterdir():
        candidate.unlink()
    (invalid_dir / canonical_file).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_explain(report=str(report_dir), baseline=str(baseline_dir))

    assert exc.value.exit_code == 2


def test_report_html_resolves_canonical_directory_input(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    report_json = report_dir / "evaluation.report.json"
    report_json.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "invarlock.cli.commands.export_html.export_html_command",
        lambda **kwargs: captured.update(kwargs),
        raising=False,
    )

    report_mod.report_html(input=str(report_dir), output="out.html")

    assert captured["input"] == str(report_json.resolve())


def test_report_html_rejects_noncanonical_directory(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "my_report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_html(input=str(report_dir), output="out.html")

    assert exc.value.exit_code == 2
