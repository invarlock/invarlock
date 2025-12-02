from __future__ import annotations

from typer.testing import CliRunner


def test_cli_top_level_help_smoke(monkeypatch):
    # Avoid heavy discovery in help smoke
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    monkeypatch.setenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY", "1")
    from invarlock.cli.app import app

    runner = CliRunner()
    for args in (
        ["--help"],
        ["certify", "--help"],
        ["report", "--help"],
        ["run", "--help"],
        ["plugins", "--help"],
        ["doctor", "--help"],
    ):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, f"help failed for: {' '.join(args)} -> {res.output}"


def test_command_wrappers_importable():
    # Ensure import surface remains stable
    from invarlock.cli.commands import (
        certify_command,
        doctor_command,
        explain_gates_command,
        export_html_command,
        plugins_command,
        report_command,
        run_command,
        verify_command,
    )

    # Basic type checks – they should be callables or Typer callbacks
    for obj in (
        certify_command,
        doctor_command,
        explain_gates_command,
        export_html_command,
        plugins_command,
        run_command,
        verify_command,
        report_command,
    ):
        assert callable(obj)
