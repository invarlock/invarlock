import os
import sys
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
import invarlock.cli.commands.plugins as plugins_mod
from invarlock.cli.app import app

runner = CliRunner()


def test_plugins_install_dry_run_extras():
    result = runner.invoke(app, ["plugins", "install", "gptq", "--dry-run"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Action: install" in out
    assert "Package: invarlock[gptq]" in out
    assert "Mode:" in out
    assert "Result: ok" in out

    result2 = runner.invoke(app, ["plugins", "install", "awq", "gpu", "--dry-run"])
    assert result2.exit_code == 0, result2.output
    out2 = result2.output
    assert "Action: install" in out2
    assert "Package: invarlock[awq] invarlock[gpu]" in out2
    assert "Mode:" in out2
    assert "Result: ok" in out2


def test_plugins_install_alias_and_unknown():
    ok = runner.invoke(app, ["plugins", "install", "hf_bnb", "--dry-run"])
    assert ok.exit_code == 0, ok.output
    assert "Package: invarlock[gpu]" in ok.output

    bad = runner.invoke(app, ["plugins", "install", "does_not_exist", "--dry-run"])
    assert bad.exit_code != 0
    assert "Action: install" in bad.output
    assert "Result: not-found" in bad.output


def test_plugins_install_aliases_onnx_and_transformers():
    result = runner.invoke(app, ["plugins", "install", "onnx", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "invarlock[onnx]" in result.output

    adapters = runner.invoke(app, ["plugins", "install", "transformers", "--dry-run"])
    assert adapters.exit_code == 0, adapters.output
    assert "invarlock[adapters]" in adapters.output


def test_plugins_install_unknown_without_explicit_dry_run(monkeypatch):
    outputs: list[str] = []

    class CaptureConsole:
        def print(self, *args, **kwargs):
            outputs.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr(plugins_mod, "console", CaptureConsole(), raising=False)
    with pytest.raises(typer.Exit) as exc:
        plugins_mod.plugins_install_command(
            ["mystery"], dry_run=False, apply=False, upgrade=False
        )
    assert exc.value.exit_code == 1
    assert any("Result: not-found" in line for line in outputs)


def test_plugins_install_apply_invokes_pip(monkeypatch):
    called: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text):
        called["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("invarlock.cli.commands.plugins.subprocess.run", fake_run)
    monkeypatch.delenv("INVARLOCK_PLUGINS_DRY_RUN", raising=False)
    plugins_mod.plugins_install_command(
        ["gptq"], upgrade=True, dry_run=False, apply=True
    )
    assert called["cmd"][0] == sys.executable
    assert called["cmd"][1:4] == ["-m", "pip", "install"]
    assert "--upgrade" in called["cmd"]
