# ruff: noqa: E402
from __future__ import annotations

import importlib
import os

from click.termui import strip_ansi
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from tests.cli.run._internal_cli import internal_run_app


def test_run_module_imports_and_exposes_run_command():
    mod = importlib.import_module("invarlock.cli.commands.run")
    assert hasattr(mod, "run_command"), "run_command symbol must be present"


def test_cli_run_help_includes_edit_label_and_metric_kind():
    result = CliRunner().invoke(internal_run_app, ["run", "--help"])
    assert result.exit_code == 0
    stdout = strip_ansi(result.stdout)
    assert "--edit-label" in stdout
    assert "--metric-kind" in stdout
    assert "quant_rtn" in stdout
    assert "quant|mixed" not in stdout


def test_cli_run_accepts_edit_label_flag():
    result = CliRunner().invoke(
        internal_run_app, ["run", "--edit-label", "noop", "--help"]
    )
    assert result.exit_code == 0
