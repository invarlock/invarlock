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


def test_plugins_uninstall_dry_run_gptq():
    result = runner.invoke(app, ["plugins", "uninstall", "gptq", "--dry-run"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Action: uninstall" in out
    assert "Package: invarlock[gptq]" in out or "Package: auto-gptq" in out
    assert "Mode:" in out
    assert ("Result: ok" in out) or ("Result: skipped" in out)


def test_plugins_uninstall_aliases():
    # AWQ alias
    result_awq = runner.invoke(app, ["plugins", "uninstall", "hf_awq", "--dry-run"])
    assert result_awq.exit_code == 0, result_awq.output
    out_awq = result_awq.output
    assert "Action: uninstall" in out_awq
    assert "Mode:" in out_awq
    assert ("Result: ok" in out_awq) or ("Result: skipped" in out_awq)

    # GPU/BnB extra expressed as extras syntax
    result_gpu = runner.invoke(
        app, ["plugins", "uninstall", "invarlock[gpu]", "--dry-run"]
    )
    assert result_gpu.exit_code == 0, result_gpu.output
    out_gpu = result_gpu.output
    assert "Action: uninstall" in out_gpu
    assert "Package: invarlock[gpu]" in out_gpu or "bitsandbytes" in out_gpu
    assert "Mode:" in out_gpu
    assert ("Result: ok" in out_gpu) or ("Result: skipped" in out_gpu)

    # ONNX alias
    result_onnx = runner.invoke(app, ["plugins", "uninstall", "onnx", "--dry-run"])
    assert result_onnx.exit_code == 0, result_onnx.output
    assert "onnxruntime" in result_onnx.output


def test_plugins_uninstall_unknown():
    result = runner.invoke(app, ["plugins", "uninstall", "does_not_exist", "--dry-run"])
    assert result.exit_code != 0
    assert "Action: uninstall" in result.output
    assert "Result: not-found" in result.output


def test_plugins_uninstall_apply_invokes_pip(monkeypatch):
    called: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text):
        called["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("invarlock.cli.commands.plugins.subprocess.run", fake_run)
    monkeypatch.delenv("INVARLOCK_PLUGINS_DRY_RUN", raising=False)
    plugins_mod.plugins_uninstall_command(["gptq"], yes=True, dry_run=False, apply=True)
    assert called["cmd"][0] == sys.executable
    assert called["cmd"][1:4] == ["-m", "pip", "uninstall"]
    assert "-y" in called["cmd"]


def test_plugins_uninstall_prompt_cancel(monkeypatch):
    outputs: list[str] = []

    class CaptureConsole:
        def print(self, *args, **kwargs):
            outputs.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr(plugins_mod, "console", CaptureConsole(), raising=False)
    monkeypatch.setattr(plugins_mod.typer, "confirm", lambda msg: False, raising=False)
    monkeypatch.delenv("INVARLOCK_PLUGINS_DRY_RUN", raising=False)
    with pytest.raises(typer.Exit) as exc:
        plugins_mod.plugins_uninstall_command(
            ["gptq"], yes=False, dry_run=False, apply=True
        )
    assert exc.value.exit_code == 0
    joined = " ".join(outputs)
    assert "Result: skipped" in joined
