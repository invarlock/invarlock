from __future__ import annotations

from click.termui import strip_ansi
from typer.testing import CliRunner


def _load_app(monkeypatch):
    # Ensure lightweight import path and skip heavy discovery
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "0")
    from invarlock.cli.app import app

    return app


def test_run_command_is_not_public(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["run"])
    assert res.exit_code == 2, res.output
    out = strip_ansi(res.output)
    assert "No such command 'run'" in out


def test_run_help_is_not_public(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["run", "--help"])
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
        ("report", ["generate", "explain", "html", "export", "validate"]),
        (
            "advanced",
            [
                "evidence-pack",
                "policy",
                "plugins",
                "calibrate",
                "runtime-behavior",
                "runtime-verify",
            ],
        ),
        ("advanced plugins", ["list", "guards", "edits", "adapters"]),
    ):
        res = runner.invoke(app, [*cmd.split(), "--help"])
        assert res.exit_code == 0, f"help failed for {cmd}: {res.output}"
        out = strip_ansi(res.stdout)
        for token in expected:
            assert token in out


def test_evidence_pack_help_is_verifier_only(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["advanced", "evidence-pack", "--help"])
    assert res.exit_code == 0, res.output
    out = strip_ansi(res.stdout)
    assert "verify" in out
    assert "inspect" in out
    assert "verify-set" in out
    assert "build" not in out
    assert "keygen" not in out
    assert "seal" not in out


def test_report_explain_help_mentions_evaluation_bundle(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["report", "explain", "--help"])
    assert res.exit_code == 0, res.output
    out = strip_ansi(res.stdout)
    normalized = " ".join(out.split())
    assert "--evaluation-report" in out
    assert "Preferred evidence" in out
    assert "explains the evaluation bundle" in normalized
    assert "directly without requiring linked raw run" in normalized


def test_report_export_help_mentions_formats_and_verify_result(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["report", "export", "--help"])
    assert res.exit_code == 0, res.output
    out = strip_ansi(res.stdout)
    assert "--format" in out
    assert "mlflow-tags" in out
    assert "model-card-md" in out
    assert "release-re" in out
    assert "--verify-result" in out


def test_runtime_verify_help_is_single_command_surface(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["advanced", "runtime-verify", "--help"])
    assert res.exit_code == 0, res.output
    out = strip_ansi(res.stdout)
    assert "COMMAND [ARGS]..." not in out
    assert "--report" in out
    assert "--manifest" in out


def test_plugin_management_subcommands_are_removed(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    for args in (
        ["advanced", "plugins", "install"],
        ["advanced", "plugins", "uninstall"],
    ):
        res = runner.invoke(app, args)
        assert res.exit_code == 2, res.output
