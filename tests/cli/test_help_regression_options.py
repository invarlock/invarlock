from __future__ import annotations

from click.termui import strip_ansi
from typer.testing import CliRunner


def _load_app(monkeypatch):
    # Ensure lightweight import path and skip heavy discovery
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "0")
    from invarlock.cli.app import app

    return app


def test_run_command_reports_migration(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["run"])
    assert res.exit_code == 2, res.output
    out = strip_ansi(res.output)
    assert "No such command 'run'" in out


def test_evaluate_help_exposes_baseline_and_subject(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["evaluate", "--help"])
    assert res.exit_code == 0, res.output
    out = strip_ansi(res.stdout)
    assert "--baseline" in out or "--baseline" in out
    assert "--subject" in out or "--subject" in out
    assert "--preset" in out


def test_doctor_help_is_typed(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["doctor", "--help"])
    assert res.exit_code == 0, res.output
    out = strip_ansi(res.stdout)
    assert "ARGS KWARGS" not in out
    # Provide a config option
    assert "--config" in out
    assert "report.json" in out
    assert "evaluation.report.json" in out


def test_groups_help_list_subcommands(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    for cmd, expected in (
        ("report", ["verify", "explain", "html", "validate"]),
        ("advanced", ["proof-pack", "policy", "plugins", "calibrate"]),
        ("advanced plugins", ["list", "guards", "edits", "adapters"]),
    ):
        res = runner.invoke(app, [*cmd.split(), "--help"])
        assert res.exit_code == 0, f"help failed for {cmd}: {res.output}"
        out = strip_ansi(res.stdout)
        for token in expected:
            assert token in out


def test_proof_pack_build_help_mentions_explicit_report_files(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["advanced", "proof-pack", "build", "--help"])
    assert res.exit_code == 0, res.output
    out = strip_ansi(res.stdout)
    assert "evaluation.report.json" in out
    assert "runtime.manifest.json" in out


def test_plugin_management_subcommands_are_removed(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    for args in (
        ["advanced", "plugins", "install"],
        ["advanced", "plugins", "uninstall"],
    ):
        res = runner.invoke(app, args)
        assert res.exit_code == 2, res.output
