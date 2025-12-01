from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import typer


def _import_run_module():
    # transformers stub to avoid heavy import via model_profile
    if "transformers" not in sys.modules:
        tr = types.ModuleType("transformers")

        class _Tok:
            pad_token = "<pad>"
            eos_token = "<eos>"

            def get_vocab(self):
                return {"<pad>": 0, "<eos>": 1}

        class _Auto:
            @staticmethod
            def from_pretrained(*a, **k):
                return _Tok()

        class _GPT2(_Auto):
            pass

        tr.AutoTokenizer = _Auto  # type: ignore[attr-defined]
        tr.GPT2Tokenizer = _GPT2  # type: ignore[attr-defined]
        sys.modules["transformers"] = tr
        sub = types.ModuleType("transformers.tokenization_utils_base")
        sub.PreTrainedTokenizerBase = object  # type: ignore[attr-defined]
        sys.modules["transformers.tokenization_utils_base"] = sub

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
    cfg.write_text("model: {id: gpt2, adapter: hf_gpt2}\n")
    import invarlock.cli.config as cfg_mod
    from invarlock.cli.errors import InvarlockError

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
    cfg.write_text("model: {id: gpt2, adapter: hf_gpt2}\n")

    def _raise_val(*a, **k):
        raise ValueError("Invalid RunReport blah")

    import invarlock.cli.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", _raise_val)
    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(config=str(cfg), device="cpu", profile=None, baseline=None)
    assert ei.value.exit_code == 2
