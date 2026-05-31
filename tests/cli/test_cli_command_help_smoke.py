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
        ["report", "--help"],
        ["verify", "--help"],
        ["doctor", "--help"],
        ["advanced", "--help"],
        ["advanced", "calibrate", "--help"],
        ["advanced", "evidence-pack", "--help"],
        ["advanced", "policy", "--help"],
        ["advanced", "plugins", "--help"],
    ):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, f"help failed for: {' '.join(args)} -> {res.output}"


def test_command_wrappers_importable():
    from invarlock.cli.commands.calibrate import calibrate_app
    from invarlock.cli.commands.doctor import doctor_command
    from invarlock.cli.commands.evaluate import evaluate_command
    from invarlock.cli.commands.evidence_pack import (
        build_command as evidence_pack_build_command,
    )
    from invarlock.cli.commands.evidence_pack import (
        inspect_command as evidence_pack_inspect_command,
    )
    from invarlock.cli.commands.evidence_pack import (
        verify_command as evidence_pack_verify_command,
    )
    from invarlock.cli.commands.explain_gates import explain_gates_command
    from invarlock.cli.commands.plugins import plugins_command
    from invarlock.cli.commands.policy import build_command as policy_build_command
    from invarlock.cli.commands.policy import verify_command as policy_verify_command
    from invarlock.cli.commands.report import export_html_command
    from invarlock.cli.commands.verify import verify_command
    from invarlock.reporting.report_contract import generate_reports

    for obj in (
        evaluate_command,
        calibrate_app,
        doctor_command,
        explain_gates_command,
        export_html_command,
        policy_build_command,
        policy_verify_command,
        evidence_pack_build_command,
        evidence_pack_inspect_command,
        evidence_pack_verify_command,
        plugins_command,
        verify_command,
        generate_reports,
    ):
        assert callable(obj)
