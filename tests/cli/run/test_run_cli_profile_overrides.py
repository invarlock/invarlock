from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import typer

from tests.conftest import install_transformers_tokenizer_stub


def _write_yaml_cfg(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def _import_run_module():
    install_transformers_tokenizer_stub()

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
          adapter: hf_causal
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
    import invarlock.cli.run_runtime_exec as runtime_mod

    monkeypatch.setattr(
        dev_mod, "validate_device_for_config", lambda *a, **k: (False, "bad device")
    )
    monkeypatch.setattr(
        runtime_mod, "detect_model_profile", lambda model_id, adapter: _StubProfile()
    )

    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(
            config=str(cfg_path),
            device="cpu",
            profile="ci",  # exercise apply_profile
            edit="quant_rtn",  # exercise canonical edit override path
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
          adapter: hf_causal    # concrete; auto-adapter path is a no-op
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
    import invarlock.cli.run_runtime_exec as runtime_mod

    monkeypatch.setattr(
        dev_mod, "validate_device_for_config", lambda *a, **k: (False, "bad device")
    )
    monkeypatch.setattr(
        runtime_mod, "detect_model_profile", lambda model_id, adapter: _StubProfile()
    )

    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(
            config=str(cfg_path),
            device="cpu",
            profile=None,
            baseline=None,
        )
    assert ei.value.exit_code == 1
