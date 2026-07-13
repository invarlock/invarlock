from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

import invarlock.cli.commands.report as report_mod
import invarlock.reporting.report_contract as report_contract_mod
import invarlock.reporting.report_schema as schema_mod
from invarlock.reporting.report_contract import ReportGenerationResult, generate_reports


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


def test_generate_reports_normalizes_all_formats(monkeypatch):
    saved = {}

    def fake_save(primary, out_dir, *, formats, **kwargs):
        saved["formats"] = formats
        return {fmt: f"{fmt}.json" for fmt in formats}

    monkeypatch.setattr(
        report_contract_mod, "load_report_payload", lambda _path: _make_primary_report()
    )
    monkeypatch.setattr(report_contract_mod, "save_report", fake_save, raising=False)

    generate_reports(run="run.json", format="all")
    assert saved["formats"] == ["json", "markdown", "html"]


def test_report_command_evaluation_report_validation_block(monkeypatch):
    result = _make_generation_result(
        evaluation_report={
            "validation": {"overall": True},
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 45.657,
                "final": 47.082,
                "ratio_vs_baseline": 1.0,
                "display_ci": [0.9981, 1.0019],
            },
        },
        validation_block={
            "overall_pass": True,
            "rows": [{"label": "primary_metric", "status": "PASS"}],
        },
    )
    captured: list[str] = []

    class _CaptureConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(report_mod, "console", _CaptureConsole())

    report_mod._render_generation_result(
        result=result,
        style="audit",
        no_color=False,
    )
    out = "\n".join(captured)
    assert "PRIMARY METRIC" in out
    assert "CI (95%)" in out
    assert "[0.998, 1.002]" in out


def test_report_command_summary_includes_total_time_suffix(monkeypatch):
    result = _make_generation_result(
        evaluation_report={"validation": {"overall": True}, "primary_metric": {}},
        validation_block={"overall_pass": True, "rows": []},
    )
    monkeypatch.setattr(report_mod, "perf_counter", lambda: 126.0)

    captured: list[str] = []

    class _CaptureConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(report_mod, "console", _CaptureConsole())

    report_mod._render_generation_result(
        result=result,
        style="audit",
        no_color=False,
        summary_baseline_seconds=1.0,
        summary_subject_seconds=2.0,
        summary_report_start=120.0,
    )
    out = "\n".join(captured)
    assert "EVALUATION REPORT SUMMARY" in out
    assert "[9.00s]" in out


def test_report_command_summary_handles_suffix_formatting_failures(monkeypatch):
    result = _make_generation_result(
        evaluation_report={"validation": {"overall": True}, "primary_metric": {}},
        validation_block={"overall_pass": True, "rows": []},
    )
    monkeypatch.setattr(report_mod, "perf_counter", lambda: 126.0)

    captured: list[str] = []

    class _CaptureConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(report_mod, "console", _CaptureConsole())

    report_mod._render_generation_result(
        result=result,
        style="audit",
        no_color=False,
        summary_baseline_seconds=1.0,
        summary_subject_seconds=2.0,
        summary_report_start="bad",
    )
    out = "\n".join(captured)
    assert "EVALUATION REPORT SUMMARY" in out
    assert "[9.00s]" not in out


def test_report_command_prints_telemetry_summary_line(monkeypatch):
    result = _make_generation_result(
        evaluation_report={"validation": {"overall": True}, "primary_metric": {}},
        validation_block={"overall_pass": True, "rows": []},
    )
    monkeypatch.setattr(report_mod, "_telemetry_output_enabled", lambda: True)
    monkeypatch.setattr(
        report_mod, "_telemetry_summary_line", lambda _report: "telemetry"
    )

    captured: list[str] = []

    class _CaptureConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(report_mod, "console", _CaptureConsole())

    report_mod._render_generation_result(result=result, style="audit", no_color=False)
    assert "telemetry" in "\n".join(captured)


def test_report_command_skips_empty_telemetry_summary_line(monkeypatch):
    result = _make_generation_result(
        evaluation_report={"validation": {"overall": True}, "primary_metric": {}},
        validation_block={"overall_pass": True, "rows": []},
    )
    monkeypatch.setattr(report_mod, "_telemetry_output_enabled", lambda: True)
    monkeypatch.setattr(report_mod, "_telemetry_summary_line", lambda _report: "")

    captured: list[str] = []

    class _CaptureConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(report_mod, "console", _CaptureConsole())

    report_mod._render_generation_result(result=result, style="audit", no_color=False)
    assert "telemetry" not in "\n".join(captured)


def test_report_command_evaluation_report_validation_error(monkeypatch):
    monkeypatch.setattr(
        report_mod,
        "load_run_report_input_json",
        lambda path: (Path(path), {"meta": {}}),
    )
    monkeypatch.setattr(
        report_mod,
        "_generate_reports",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bad report")),
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *args, **kwargs: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_callback(
            type("Ctx", (), {"resilient_parsing": False, "invoked_subcommand": None})(),
            run="run.json",
            format="report",
            compare=None,
            baseline="baseline.json",
            output=None,
            style="audit",
            no_color=False,
        )
    assert exc.value.exit_code == 1


def test_report_command_renders_non_report_artifacts_without_summary(monkeypatch):
    result = _make_generation_result(
        formats=["json"],
        saved_files={"json": "report.json", "html": "report.html"},
        evaluation_report=None,
        validation_block=None,
    )
    captured: list[str] = []

    class _CaptureConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(report_mod, "console", _CaptureConsole())

    report_mod._render_generation_result(result=result, style="audit", no_color=False)
    rendered = "\n".join(captured)
    assert "EVALUATION REPORT SUMMARY" not in rendered
    assert "Output" in rendered
    assert "report.json" in rendered
    assert "report.html" in rendered


def test_report_command_rejects_noncanonical_run_directory(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "my_report.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *args, **kwargs: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_callback(
            type("Ctx", (), {"resilient_parsing": False, "invoked_subcommand": None})(),
            run=str(report_dir),
            format="json",
            compare=None,
            baseline=None,
            output=None,
            style="audit",
            no_color=False,
        )

    assert exc.value.exit_code == 2


def test_report_command_maps_nonbaseline_value_error_to_exit_two(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        report_mod,
        "load_run_report_input_json",
        lambda path: (Path(path), {"meta": {}}),
    )

    monkeypatch.setattr(
        report_mod,
        "_generate_reports",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad input path")),
    )
    monkeypatch.setattr(
        report_mod,
        "_raise_report_input_failure",
        lambda message, *, no_color=False: (
            captured.update({"message": message, "no_color": no_color}),
            (_ for _ in ()).throw(typer.Exit(2)),
        )[1],
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_callback(
            type("Ctx", (), {"resilient_parsing": False, "invoked_subcommand": None})(),
            run="run.json",
            format="json",
            compare=None,
            baseline=None,
            output=None,
            style="audit",
            no_color=False,
        )

    assert exc.value.exit_code == 2
    assert captured == {"message": "bad input path", "no_color": False}


def test_report_callback_maps_generic_error_to_exit_one(monkeypatch) -> None:
    monkeypatch.setattr(
        report_mod,
        "load_run_report_input_json",
        lambda path: (Path(path), {"meta": {}}),
    )
    monkeypatch.setattr(
        report_mod,
        "_generate_reports",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_callback(
            type("Ctx", (), {"resilient_parsing": False, "invoked_subcommand": None})(),
            run="run.json",
            format="json",
            compare=None,
            baseline=None,
            output=None,
            style="audit",
            no_color=False,
        )

    assert exc.value.exit_code == 1


def test_artifact_entries_includes_unordered_extra_saved_files() -> None:
    entries = report_mod._artifact_entries(
        {"json": "a.json", "telemetry": "telemetry.json"},
        "out",
    )

    assert ("Output", "out") in entries
    assert ("JSON", "a.json") in entries
    assert ("TELEMETRY", "telemetry.json") in entries


def test_fmt_ci_95_rejects_non_pair_values() -> None:
    assert report_mod._fmt_ci_95("not-a-pair") is None
    assert report_mod._fmt_ci_95(None) is None


def test_report_command_summary_skips_telemetry_when_disabled(monkeypatch) -> None:
    result = _make_generation_result(
        primary={"meta": {}, "edit": {}, "metrics": {}},
        evaluation_report={"primary_metric": None},
        validation_block={"overall_pass": False, "rows": []},
    )
    captured: list[str] = []

    class _CaptureConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(report_mod, "console", _CaptureConsole())
    monkeypatch.setattr(report_mod, "_telemetry_output_enabled", lambda: False)
    monkeypatch.setattr(
        report_mod,
        "_telemetry_summary_line",
        lambda _report: (_ for _ in ()).throw(RuntimeError("should not run")),
    )

    report_mod._render_generation_result(result=result, style="audit", no_color=False)
    rendered = "\n".join(captured)
    assert "No validation rows" in rendered
    assert "Status         Unavailable" in rendered


def test_fmt_ci_95_rejects_non_finite_and_non_numeric_bounds() -> None:
    assert report_mod._fmt_ci_95(["bad", 1.0]) is None
    assert report_mod._fmt_ci_95([1.0, float("inf")]) is None


def test_report_command_summary_handles_missing_optional_fields(monkeypatch) -> None:
    result = _make_generation_result(
        primary={"meta": {}, "edit": {}, "metrics": {}, "_sentinel": True},
        evaluation_report={"validation": {}},
        validation_block={"overall_pass": False, "rows": "not-a-list"},
    )
    captured: list[str] = []

    class _CaptureConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(report_mod, "console", _CaptureConsole())
    monkeypatch.setattr(report_mod, "_telemetry_output_enabled", lambda: True)
    monkeypatch.setattr(report_mod, "_telemetry_summary_line", lambda _report: None)

    report_mod._render_generation_result(result=result, style="audit", no_color=False)
    rendered = "\n".join(captured)
    assert "No validation rows" in rendered
    assert "Status         Unavailable" in rendered
    assert "Run ID" not in rendered
    assert "Model" not in rendered
    assert "Edit" not in rendered


def test_render_generation_result_maps_report_input_error_to_exit_two(monkeypatch):
    result = _make_generation_result(
        formats=["json"],
        saved_files={"json": "report.json"},
        evaluation_report=None,
        validation_block=None,
    )
    captured: dict[str, object] = {}

    def _raise_input(message: str, *, no_color: bool = False) -> None:
        captured["message"] = message
        captured["no_color"] = no_color
        raise typer.Exit(2)

    monkeypatch.setattr(
        report_mod,
        "print_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            report_mod.ReportInputError("invalid_report_input", Path("bad.json"))
        ),
    )
    monkeypatch.setattr(report_mod, "_raise_report_input_failure", _raise_input)

    with pytest.raises(typer.Exit) as exc:
        report_mod._render_generation_result(
            result=result, style="audit", no_color=True
        )

    assert exc.value.exit_code == 2
    assert captured == {
        "message": "Invalid report input: bad.json",
        "no_color": True,
    }


def test_render_generation_result_maps_value_error_to_exit_two(monkeypatch):
    result = _make_generation_result(
        formats=["json"],
        saved_files={"json": "report.json"},
        evaluation_report=None,
        validation_block=None,
    )
    monkeypatch.setattr(
        report_mod,
        "_artifact_entries",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad value")),
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *args, **kwargs: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod._render_generation_result(
            result=result, style="audit", no_color=False
        )
    assert exc.value.exit_code == 2


def test_render_generation_result_maps_unexpected_error_to_exit_one(monkeypatch):
    result = _make_generation_result(
        formats=["json"],
        saved_files={"json": "report.json"},
        evaluation_report=None,
        validation_block=None,
    )
    monkeypatch.setattr(
        report_mod,
        "_artifact_entries",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *args, **kwargs: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod._render_generation_result(
            result=result, style="audit", no_color=False
        )
    assert exc.value.exit_code == 1


def test_render_generation_result_maps_inner_report_error_to_exit_one(monkeypatch):
    result = _make_generation_result(
        evaluation_report={"primary_metric": {}},
        validation_block={"overall_pass": True, "rows": []},
    )
    monkeypatch.setattr(report_mod, "_telemetry_output_enabled", lambda: True)
    monkeypatch.setattr(
        report_mod,
        "_telemetry_summary_line",
        lambda _report: (_ for _ in ()).throw(RuntimeError("bad telemetry")),
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *args, **kwargs: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod._render_generation_result(
            result=result, style="audit", no_color=False
        )
    assert exc.value.exit_code == 1


def test_render_generation_result_maps_console_failures_to_exit_one(monkeypatch):
    result = _make_generation_result(
        evaluation_report={"primary_metric": {}},
        validation_block={"overall_pass": True, "rows": []},
    )
    monkeypatch.setattr(report_mod, "print_event", lambda *_args, **_kwargs: None)

    class _ExplodingConsole:
        def __init__(self) -> None:
            self.calls = 0

        def print(self, *_args: object, **_kwargs: object) -> None:
            self.calls += 1
            if self.calls >= 2:
                raise RuntimeError("console boom")

    monkeypatch.setattr(report_mod, "console", _ExplodingConsole())

    with pytest.raises(typer.Exit) as exc:
        report_mod._render_generation_result(
            result=result, style="audit", no_color=False
        )
    assert exc.value.exit_code == 1


def test_report_callback_skips_subcommand_and_delegates_success(monkeypatch) -> None:
    delegated: list[dict[str, object]] = []

    def fake_generate(**kwargs: object) -> ReportGenerationResult:
        delegated.append(kwargs)
        return _make_generation_result(formats=["json"], saved_files={"json": "a.json"})

    monkeypatch.setattr(report_mod, "_generate_reports", fake_generate)
    monkeypatch.setattr(
        report_mod,
        "load_run_report_input_json",
        lambda path: (Path(path), {"meta": {}}),
    )

    ctx_skip = type(
        "Ctx", (), {"resilient_parsing": False, "invoked_subcommand": "verify"}
    )()
    report_mod.report_callback(
        ctx_skip,
        run="run.json",
        format="json",
        compare=None,
        baseline=None,
        output=None,
        style="audit",
        no_color=False,
    )
    assert delegated == []

    ctx_run = type(
        "Ctx", (), {"resilient_parsing": False, "invoked_subcommand": None}
    )()
    report_mod.report_callback(
        ctx_run,
        run="run.json",
        format="json",
        compare="compare.json",
        baseline="baseline.json",
        output="out",
        style="friendly",
        no_color=True,
    )
    assert delegated == [
        {
            "run": "run.json",
            "format": "json",
            "compare": "compare.json",
            "baseline": "baseline.json",
            "output": "out",
        }
    ]


def test_report_command_maps_report_input_error_to_exit_two(monkeypatch) -> None:
    monkeypatch.setattr(
        report_mod,
        "load_run_report_input_json",
        lambda path: (Path(path), {"meta": {}}),
    )
    monkeypatch.setattr(
        report_mod,
        "_generate_reports",
        lambda **_kwargs: (_ for _ in ()).throw(
            report_mod.ReportInputError("not_found", Path("run.json"))
        ),
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *args, **kwargs: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_callback(
            type("Ctx", (), {"resilient_parsing": False, "invoked_subcommand": None})(),
            run="run.json",
            format="json",
            compare=None,
            baseline=None,
            output=None,
            style="audit",
            no_color=False,
        )

    assert exc.value.exit_code == 2


def test_report_validate_success(monkeypatch, tmp_path):
    report = tmp_path / "evaluation.report.json"
    report.write_text(
        json.dumps(_minimal_evaluation_report_payload()), encoding="utf-8"
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )
    monkeypatch.setattr(
        schema_mod,
        "validate_report",
        lambda payload: True,
        raising=False,
    )
    report_mod.report_validate(report=str(report))
    assert report.is_file()


def test_report_validate_accepts_canonical_directory(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "evaluation.report.json").write_text(
        json.dumps(_minimal_evaluation_report_payload()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )
    monkeypatch.setattr(
        schema_mod,
        "validate_report",
        lambda payload: True,
        raising=False,
    )

    report_mod.report_validate(report=str(report_dir))
    assert (report_dir / "evaluation.report.json").is_file()


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


def test_report_validate_rejects_ambiguous_directory(monkeypatch, tmp_path):
    report_dir = tmp_path / "report-dir"
    report_dir.mkdir()
    (report_dir / "report.json").write_text("{}", encoding="utf-8")
    (report_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )

    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(report_dir))

    assert exc.value.exit_code == 2


def test_report_validate_schema_failure(monkeypatch, tmp_path):
    report = tmp_path / "evaluation.report.json"
    report.write_text(
        json.dumps(_minimal_evaluation_report_payload()), encoding="utf-8"
    )
    monkeypatch.setattr(
        report_mod, "console", type("C", (), {"print": lambda *_: None})()
    )
    monkeypatch.setattr(
        schema_mod,
        "validate_report",
        lambda payload: False,
        raising=False,
    )
    with pytest.raises(typer.Exit) as exc:
        report_mod.report_validate(report=str(report))
    assert exc.value.exit_code == 2
