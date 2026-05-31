from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

import invarlock.cli.commands.report as report_mod
import invarlock.reporting.report_schema as schema_mod
from invarlock.reporting.report_contract import ReportGenerationResult


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


def _make_generation_result(
    *,
    primary: dict | None = None,
    baseline: dict | None = None,
    output_dir: str = "out",
    saved_files: dict[str, str] | None = None,
    evaluation_report: dict | None = None,
    validation_block: dict | None = None,
    formats: list[str] | None = None,
) -> ReportGenerationResult:
    return ReportGenerationResult(
        output_dir=output_dir,
        formats=formats or ["report"],
        saved_files=saved_files or {"report": "evaluation.report.json"},
        primary_report=primary or _make_primary_report(),
        compare_report=None,
        baseline_report=baseline or _make_primary_report(),
        evaluation_report=evaluation_report,
        validation_block=validation_block,
    )


def _minimal_evaluation_report_payload() -> dict[str, object]:
    return {"schema_version": "v1", "validation": {}}


def test_report_validate_value_error(monkeypatch, tmp_path):
    report = tmp_path / "evaluation.report.json"
    report.write_text(
        json.dumps(_minimal_evaluation_report_payload()), encoding="utf-8"
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    def _raise_val(payload):
        raise ValueError("bad schema")

    monkeypatch.setattr(schema_mod, "validate_report", _raise_val, raising=False)
    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(report))
    assert exc.value.exit_code == 2


def test_report_validate_generic_error(monkeypatch, tmp_path):
    report = tmp_path / "evaluation.report.json"
    report.write_text(
        json.dumps(_minimal_evaluation_report_payload()), encoding="utf-8"
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    def _raise(payload):
        raise RuntimeError("boom")

    monkeypatch.setattr(schema_mod, "validate_report", _raise, raising=False)
    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(report))
    assert exc.value.exit_code == 1


def test_report_validate_read_failure(monkeypatch, tmp_path):
    report = tmp_path / "evaluation.report.json"
    report.write_text(
        json.dumps(_minimal_evaluation_report_payload()), encoding="utf-8"
    )
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


def test_report_explain_resolves_canonical_directories(monkeypatch, tmp_path):
    run_report = {
        "meta": {"seed": 1},
        "data": {},
        "edit": {},
        "guards": [],
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 1.0}},
        "artifacts": {},
        "flags": {},
    }
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    report_json = report_dir / "report.json"
    report_json.write_text(json.dumps(run_report), encoding="utf-8")
    baseline_dir = tmp_path / "baseline-dir"
    baseline_dir.mkdir()
    baseline_json = baseline_dir / "report.json"
    baseline_json.write_text(json.dumps(run_report), encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "invarlock.cli.commands.explain_gates.explain_gates_command",
        lambda **kwargs: captured.update(kwargs),
        raising=False,
    )

    report_mod.report_explain(
        subject_report=str(report_dir),
        baseline_report=str(baseline_dir),
    )

    assert captured["subject_report"] == str(report_json.resolve())
    assert captured["baseline_report"] == str(baseline_json.resolve())


def test_report_explain_resolves_linked_run_reports_from_evaluation_bundle(
    monkeypatch, tmp_path
):
    run_report = {
        "meta": {"seed": 1},
        "data": {},
        "edit": {},
        "guards": [],
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 1.0}},
        "artifacts": {},
        "flags": {},
    }
    subject_dir = tmp_path / "runs" / "subject"
    baseline_dir = tmp_path / "runs" / "baseline"
    subject_dir.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)
    subject_json = subject_dir / "report.json"
    baseline_json = baseline_dir / "report.json"
    subject_json.write_text(json.dumps(run_report), encoding="utf-8")
    baseline_json.write_text(json.dumps(run_report), encoding="utf-8")

    evaluation_dir = tmp_path / "reports" / "eval"
    evaluation_dir.mkdir(parents=True)
    evaluation_json = evaluation_dir / "evaluation.report.json"
    evaluation_json.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "validation": {},
                "provenance": {
                    "edited": {"report_path": "../../runs/subject/report.json"},
                    "baseline": {"report_path": "../../runs/baseline/report.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "invarlock.cli.commands.explain_gates.explain_gates_command",
        lambda **kwargs: captured.update(kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    report_mod.report_explain(evaluation_report=str(evaluation_json))

    assert captured["subject_report"] == str(subject_json.resolve())
    assert captured["baseline_report"] == str(baseline_json.resolve())


@pytest.mark.parametrize("invalid_slot", ["report", "baseline"])
def test_report_explain_rejects_evaluation_report_bundle(
    monkeypatch, tmp_path, invalid_slot
):
    run_report = {
        "meta": {"seed": 1},
        "data": {},
        "edit": {},
        "guards": [],
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 1.0}},
        "artifacts": {},
        "flags": {},
    }
    report_json = tmp_path / "report.json"
    report_json.write_text(json.dumps(run_report), encoding="utf-8")
    baseline_json = tmp_path / "baseline-report.json"
    baseline_json.write_text(json.dumps(run_report), encoding="utf-8")
    evaluation_json = tmp_path / "evaluation.report.json"
    evaluation_json.write_text(
        json.dumps({"schema_version": "v1", "validation": {}}), encoding="utf-8"
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "invarlock.cli.commands.explain_gates.explain_gates_command",
        lambda **kwargs: captured.update(kwargs),
        raising=False,
    )
    if invalid_slot == "report":
        report_json = evaluation_json
    else:
        baseline_json = evaluation_json
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_explain(
            subject_report=str(report_json),
            baseline_report=str(baseline_json),
        )

    assert exc.value.exit_code == 2
    assert captured == {}


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
        report_mod.report_explain(
            subject_report=str(report_dir),
            baseline_report=str(baseline_dir),
        )

    assert exc.value.exit_code == 2


@pytest.mark.parametrize("invalid_slot", ["report", "baseline"])
def test_report_explain_rejects_ambiguous_directory_inputs(
    monkeypatch, tmp_path, invalid_slot
):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "report.json").write_text("{}", encoding="utf-8")
    baseline_dir = tmp_path / "baseline-dir"
    baseline_dir.mkdir()
    (baseline_dir / "report.json").write_text("{}", encoding="utf-8")
    invalid_dir = report_dir if invalid_slot == "report" else baseline_dir
    (invalid_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_explain(
            subject_report=str(report_dir),
            baseline_report=str(baseline_dir),
        )

    assert exc.value.exit_code == 2


def test_report_html_resolves_canonical_directory_input(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    report_json = report_dir / "evaluation.report.json"
    report_json.write_text(
        json.dumps(_minimal_evaluation_report_payload()), encoding="utf-8"
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "invarlock.cli.commands.report.export_html_command",
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


def test_report_html_rejects_ambiguous_directory(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "report.json").write_text("{}", encoding="utf-8")
    (report_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_html(input=str(report_dir), output="out.html")

    assert exc.value.exit_code == 2
