from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.cli.commands import report as report_mod


def test_report_requires_run_flag_when_no_subcommand():
    r = CliRunner().invoke(app, ["report"])  # missing --run
    assert r.exit_code == 2
    assert "--run is required" in r.stdout


def test_report_cert_requires_baseline(tmp_path: Path):
    # Minimal run report path
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {"meta": {"model_id": "m", "adapter": "hf", "seed": 0, "device": "cpu"}}
        )
    )
    r = CliRunner().invoke(
        app, ["report", "--run", str(run), "--format", "report"]
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


def test_generate_reports_rejects_unknown_format():
    with patch.object(report_mod, "_load_run_report", return_value={"meta": {}}):
        with pytest.raises(report_mod.typer.Exit) as excinfo:
            report_mod._generate_reports(run="ignored.json", format="bogus")
    assert excinfo.value.exit_code == 1


def test_generate_reports_normalizes_md_and_handles_sparse_primary_metric(
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
        patch.object(
            report_mod,
            "_load_run_report",
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
        patch.object(
            report_mod,
            "_save_report",
            return_value={"report": str(tmp_path / "report.json")},
        ) as save_report,
        patch.object(
            report_mod.report_builder,
            "make_report",
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
        patch.object(report_mod.report_builder, "validate_report"),
        patch.dict(sys.modules, {"invarlock.reporting.render": render_mod}),
        patch.object(report_mod.console, "print"),
    ):
        report_mod._generate_reports(
            run=str(run_path),
            format="report",
            baseline=str(run_path),
            output=str(tmp_path / "out"),
        )
        report_mod._generate_reports(
            run=str(run_path),
            format="md",
            output=str(tmp_path / "out-md"),
        )

    first_formats = save_report.call_args_list[0].kwargs["formats"]
    second_formats = save_report.call_args_list[1].kwargs["formats"]
    assert first_formats == ["report"]
    assert second_formats == ["markdown"]
