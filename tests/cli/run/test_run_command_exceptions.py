from __future__ import annotations

from pathlib import Path

import pytest
import typer

from tests.cli._support_transformers import install_transformers_tokenizer_stub


def _import_run_module():
    install_transformers_tokenizer_stub()

    import importlib

    return importlib.import_module("invarlock.cli.commands.run")


def test_run_command_missing_config(tmp_path: Path) -> None:
    run_mod = _import_run_module()
    missing = tmp_path / "nope.yaml"
    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(
            config=str(missing), device="cpu", profile=None, baseline=None
        )
    assert ei.value.exit_code == 1


def test_run_command_invarlock_error_in_ci(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_mod = _import_run_module()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: {id: gpt2, adapter: hf_causal}\n")
    import invarlock.core.config_loader as cfg_mod
    from invarlock.core.exceptions import InvarlockError

    def _raise_invarlock(*a, **k):
        raise InvarlockError(code="E001", message="boom")

    monkeypatch.setattr(cfg_mod, "load_config", _raise_invarlock)
    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(config=str(cfg), device="cpu", profile="ci", baseline=None)
    assert ei.value.exit_code == 3


def test_run_command_schema_invalid_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_mod = _import_run_module()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: {id: gpt2, adapter: hf_causal}\n")

    def _raise_val(*a, **k):
        raise ValueError("Invalid RunReport blah")

    import invarlock.core.config_loader as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", _raise_val)
    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(config=str(cfg), device="cpu", profile=None, baseline=None)
    assert ei.value.exit_code == 2
