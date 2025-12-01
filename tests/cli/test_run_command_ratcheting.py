from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import typer


def _write_yaml_cfg(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


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


class _StubProfile:
    family = "gpt2"
    default_loss = "causal"
    default_metric = "ppl_causal"
    default_provider = "wikitext2"
    module_selectors = {}
    invariants = ()
    cert_lints = ()


def test_run_cli_profile_and_edit_override_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _write_yaml_cfg(
        tmp_path / "cfg.yaml",
        """
        model:
          id: gpt2
          adapter: hf_gpt2
        dataset:
          provider: wikitext2
          seq_len: 8
          stride: 8
        guards:
          order: []
        output:
          dir: runs
        """,
    )
    run_mod = _import_run_module()
    # Force device validation failure so we exit shortly after the profile/edit paths
    import invarlock.cli.device as dev_mod

    monkeypatch.setattr(
        dev_mod, "validate_device_for_config", lambda *a, **k: (False, "bad device")
    )
    monkeypatch.setattr(
        run_mod, "detect_model_profile", lambda model_id, adapter: _StubProfile()
    )

    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(
            config=str(cfg_path),
            device="cpu",
            profile="ci",  # exercise apply_profile
            edit="quant",  # exercise resolve_edit_kind/apply_edit_override
            tier="balanced",  # exercise auto overrides path
            probes=2,
            baseline=None,
        )
    assert ei.value.exit_code == 1


def test_run_cli_adapter_auto_noop_then_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _write_yaml_cfg(
        tmp_path / "cfg.yaml",
        """
        model:
          id: gpt2
          adapter: hf_gpt2    # concrete; auto-adapter path is a no-op
        edit:
          name: quant_rtn
        dataset:
          provider: wikitext2
          seq_len: 8
          stride: 8
        guards:
          order: []
        output:
          dir: runs
        """,
    )
    run_mod = _import_run_module()
    # Force device validation failure so we exit after adapter_auto check
    import invarlock.cli.device as dev_mod

    monkeypatch.setattr(
        dev_mod, "validate_device_for_config", lambda *a, **k: (False, "bad device")
    )
    monkeypatch.setattr(
        run_mod, "detect_model_profile", lambda model_id, adapter: _StubProfile()
    )

    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(
            config=str(cfg_path),
            device="cpu",
            profile=None,
            baseline=None,
        )
    assert ei.value.exit_code == 1
