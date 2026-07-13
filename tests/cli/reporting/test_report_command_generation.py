from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.cli.commands import report as report_mod
from invarlock.reporting import report_summary as console_mod
from invarlock.reporting.report_contract import ReportGenerationResult, generate_reports
from invarlock.reporting.report_types import create_empty_report


def test_report_requires_run_flag_when_no_subcommand():
    r = CliRunner().invoke(app, ["report"])  # missing --run
    assert r.exit_code == 0
    assert "generate" in r.stdout


def test_report_cert_requires_baseline(tmp_path: Path):
    report = create_empty_report()
    report["meta"].update({"model_id": "m", "adapter": "hf", "seed": 0})
    report["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 1.0,
        "final": 1.0,
    }
    run = tmp_path / "run.json"
    run.write_text(json.dumps(report))
    r = CliRunner().invoke(
        app, ["report", "generate", "--run", str(run), "--format", "report"]
    )  # no --baseline
    assert r.exit_code == 1
    assert "Evaluation report format requires --baseline" in r.stdout


def test_report_helpers_cover_invalid_and_long_formatting_cases():
    assert (
        report_mod._format_section_title(
            "TITLE", suffix="x" * (report_mod.SECTION_WIDTH + 5)
        )
        == f"TITLE {'x' * (report_mod.SECTION_WIDTH + 5)}"
    )
    assert report_mod._fmt_metric_value("bad") == "N/A"
    assert report_mod._fmt_metric_value(float("inf")) == "N/A"
    assert report_mod._fmt_ci_95(("bad", 1.0)) is None
    assert report_mod._fmt_ci_95((1.0, float("nan"))) is None


def test_report_command_rejects_unknown_format():
    with patch(
        "invarlock.reporting.report_contract.load_report_payload",
        return_value={"meta": {}},
    ):
        with pytest.raises(ValueError, match="Unknown --format"):
            generate_reports(run="ignored.json", format="bogus")


def test_report_command_normalizes_md_and_handles_sparse_primary_metric(
    tmp_path: Path,
) -> None:
    render_mod = types.SimpleNamespace(
        compute_console_validation_block=lambda _report: {
            "overall_pass": True,
            "rows": [],
        }
    )
    run_path = tmp_path / "run.json"
    run_path.write_text("{}", encoding="utf-8")

    with (
        patch(
            "invarlock.reporting.report_contract.load_report_payload",
            side_effect=[
                {
                    "meta": {},
                    "edit": {},
                    "metrics": {"primary_metric": {"kind": "loss", "preview": "0.5"}},
                },
                {"meta": {}, "metrics": {}},
                {"meta": {}, "metrics": {}},
            ],
        ),
        patch(
            "invarlock.reporting.report_contract.save_report",
            return_value={"markdown": str(tmp_path / "report.md")},
        ) as save_report,
        patch(
            "invarlock.reporting.report_contract.save_evaluation_bundle",
            return_value={"report": tmp_path / "report.json"},
        ) as save_bundle,
        patch(
            "invarlock.reporting.report_contract.make_report",
            return_value={
                "schema_version": "1",
                "primary_metric": {
                    "kind": "loss",
                    "preview": "0.5",
                    "final": None,
                    "ratio_vs_baseline": None,
                    "display_ci": ("bad", "ci"),
                },
            },
        ),
        patch("invarlock.reporting.report_contract.validate_report", return_value=True),
        patch(
            "invarlock.reporting.report_contract.compute_console_validation_block",
            render_mod.compute_console_validation_block,
        ),
    ):
        generate_reports(
            run=str(run_path),
            format="report",
            baseline=str(run_path),
            output=str(tmp_path / "out"),
        )
        generate_reports(
            run=str(run_path),
            format="md",
            output=str(tmp_path / "out-md"),
        )

    assert save_bundle.call_count == 1
    assert save_report.call_args_list[0].kwargs["formats"] == ["markdown"]


def test_generate_reports_fails_closed_when_evaluation_report_is_invalid(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text("{}", encoding="utf-8")

    with (
        patch(
            "invarlock.reporting.report_contract.load_report_payload",
            side_effect=[
                {"meta": {}, "metrics": {"primary_metric": {"kind": "accuracy"}}},
                {"meta": {}, "metrics": {"primary_metric": {"kind": "accuracy"}}},
            ],
        ),
        patch(
            "invarlock.reporting.report_contract.make_report",
            return_value={
                "schema_version": "1",
                "primary_metric": {
                    "kind": "accuracy",
                    "preview": 1.0,
                    "final": 1.0,
                    "delta_vs_baseline_pp": 0.0,
                },
            },
        ),
        patch(
            "invarlock.reporting.report_contract.validate_report", return_value=False
        ),
    ):
        with pytest.raises(
            ValueError, match="Generated evaluation report failed schema validation"
        ):
            generate_reports(
                run=str(run_path),
                format="report",
                baseline=str(run_path),
                output=str(tmp_path / "out"),
            )


def test_report_command_generic_failure_maps_to_exit_one(monkeypatch) -> None:
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
    monkeypatch.setattr(report_mod, "print_event", lambda *_args, **_kwargs: None)

    with pytest.raises(report_mod.typer.Exit) as excinfo:
        report_mod.report_callback(
            type("Ctx", (), {"resilient_parsing": False, "invoked_subcommand": None})(),
            run="ignored.json",
            format="json",
            compare=None,
            baseline=None,
            output=None,
            style="audit",
            no_color=False,
        )

    assert excinfo.value.exit_code == 1


def test_report_command_summary_prints_metadata_and_primary_metric_fields(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text("{}", encoding="utf-8")
    primary_report = {
        "meta": {"model_id": "subject-model", "run_id": "subject-run"},
        "edit": {"name": "quant_rtn"},
        "metrics": {},
    }
    baseline_report = {"meta": {}, "metrics": {}}
    captured: list[str] = []

    class _CaptureConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            captured.append(" ".join(str(arg) for arg in args))

    result = ReportGenerationResult(
        output_dir=str(tmp_path / "out"),
        formats=["report"],
        saved_files={"report": str(tmp_path / "evaluation.report.json")},
        primary_report=primary_report,
        compare_report=None,
        baseline_report=baseline_report,
        evaluation_report={
            "schema_version": "1",
            "run_id": "evaluation-run",
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.0,
                "final": 2.0,
                "ratio_vs_baseline": 1.5,
                "display_ci": [1.25, 1.75],
            },
        },
        validation_block=console_mod.compute_console_validation_block({}),
    )
    with patch.object(report_mod, "console", _CaptureConsole()):
        report_mod._render_generation_result(
            result=result,
            style="audit",
            no_color=False,
        )

    rendered = "\n".join(captured)
    assert "Schema Version" in rendered
    assert "Run ID" in rendered
    assert "Model" in rendered
    assert "Edit" in rendered
    assert "Preview" in rendered
    assert "Final" in rendered
    assert "Ratio" in rendered
    assert "CI (95%)" in rendered
