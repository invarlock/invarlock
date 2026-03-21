from __future__ import annotations

from typer.testing import CliRunner


def test_cli_top_level_help_smoke(monkeypatch):
    # Avoid heavy discovery in help smoke
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "0")
    from invarlock.cli.app import app

    runner = CliRunner()
    for args in (
        ["--help"],
        ["evaluate", "--help"],
        ["calibrate", "--help"],
        ["report", "--help"],
        ["proof-pack", "--help"],
        ["run", "--help"],
        ["policy", "--help"],
        ["plugins", "--help"],
        ["plugins", "install", "--help"],
        ["plugins", "uninstall", "--help"],
        ["doctor", "--help"],
    ):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, f"help failed for: {' '.join(args)} -> {res.output}"


def test_command_wrappers_importable():
    # Ensure import surface remains stable
    from invarlock.cli.commands import (
        doctor_command,
        evaluate_command,
        explain_gates_command,
        export_html_command,
        plugins_command,
        policy_build_command,
        policy_verify_command,
        proof_pack_build_command,
        proof_pack_inspect_command,
        proof_pack_verify_command,
        report_command,
        run_command,
        verify_command,
    )
    from invarlock.cli.commands.calibrate import calibrate_app

    # Basic type checks – they should be callables or Typer callbacks
    for obj in (
        evaluate_command,
        calibrate_app,
        doctor_command,
        explain_gates_command,
        export_html_command,
        policy_build_command,
        policy_verify_command,
        proof_pack_build_command,
        proof_pack_inspect_command,
        proof_pack_verify_command,
        plugins_command,
        run_command,
        verify_command,
        report_command,
    ):
        assert callable(obj)
