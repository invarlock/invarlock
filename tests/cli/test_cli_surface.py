from __future__ import annotations

from typer.testing import CliRunner

from invarlock.cli.app import app

RUNNER = CliRunner()


def _help(*args: str) -> str:
    result = RUNNER.invoke(app, [*args, "--help"])
    assert result.exit_code == 0, result.output
    return result.output


def test_root_teaches_only_the_core_user_journey() -> None:
    help_text = _help()

    for command in ("evaluate", "verify", "report"):
        assert command in help_text
    for retired in ("doctor", "advanced", "calibrate", "plugins", "inputs"):
        assert retired not in help_text


def test_evaluate_accepts_one_request_instead_of_model_flag_sprawl() -> None:
    help_text = _help("evaluate")

    assert "REQUEST" in help_text
    assert "--baseline " not in help_text
    assert "--subject " not in help_text
    assert "--edit-config" not in help_text
    assert "--clean-selection" not in help_text
    assert "--allow-network" not in help_text
    assert "--allow-remote-code" not in help_text
    assert "--allow-installed-scorers" in help_text
    assert "--preflight" in help_text


def test_verify_uses_the_bundle_and_independent_trust_anchors() -> None:
    help_text = _help("verify")

    assert "EVIDENCE" in help_text
    assert "--policy" in help_text
    assert "--expected-baseline-runtime" in help_text
    assert "--expected-subject-runtime" in help_text
    assert "--expected-baseline-artifact" in help_text
    assert "--expected-subject-artifact" in help_text
    assert "--expected-schedule" in help_text
    assert "--expected-signer" in help_text
    assert "--receipt" in help_text
    assert "--verifier-signing-key" in help_text
    assert "--verifier-identity" in help_text
    assert "--allow-installed-scorers" in help_text
    assert "--trust-profile" in help_text
    assert "--baseline" not in help_text


def test_report_renders_directly_from_the_bundle() -> None:
    help_text = _help("report")

    assert "EVIDENCE" in help_text
    assert "--html" in help_text
    assert "--run" not in help_text
    assert "generate" not in help_text
    assert "validate" not in help_text
