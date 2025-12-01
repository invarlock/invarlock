from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer


def _import_run_module():
    # transformers stub to avoid heavy import during run module import
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


def _write_yaml_cfg(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


class _StubRegistry:
    def get_adapter(self, name: str):  # minimal adapter instance
        return SimpleNamespace(name=name)

    def get_plugin_metadata(self, name: str, typ: str):  # minimal metadata
        return {"name": name, "type": typ, "module": "stub"}

    def get_edit(self, name: str):  # never reached in failing path
        return SimpleNamespace(name=name)

    def get_guard(self, name: str):  # never reached
        return SimpleNamespace(name=name)


class _StubProfile:
    family = "gpt2"
    default_loss = "causal"
    default_metric = "ppl_causal"
    default_provider = "wikitext2"
    module_selectors = {}
    invariants = ()
    cert_lints = ()


def test_run_edit_name_missing_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Minimal config without edit.name
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
    # Stub out registry and model_profile to avoid heavy imports
    run_mod = _import_run_module()
    monkeypatch.setattr(run_mod, "get_registry", lambda: _StubRegistry())
    monkeypatch.setattr(
        run_mod, "detect_model_profile", lambda model_id, adapter: _StubProfile()
    )

    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(
            config=str(cfg_path), device="cpu", profile=None, baseline=None
        )
    assert ei.value.exit_code == 1


def test_run_command_missing_torch_shows_extra_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # Simulate an environment where torch is not installed.
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[override]
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    cfg_path = _write_yaml_cfg(
        tmp_path / "cfg.yaml",
        """
        model:
          id: gpt2
          adapter: hf_gpt2
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
    # Stub profile to avoid heavy imports; we should fail before using it.
    monkeypatch.setattr(
        run_mod, "detect_model_profile", lambda model_id, adapter: _StubProfile()
    )

    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(
            config=str(cfg_path), device="cpu", profile=None, baseline=None
        )

    assert ei.value.exit_code == 1
    out = capsys.readouterr().out
    assert "Torch is required for this command." in out
    assert "invarlock[hf]" in out
    assert "invarlock[adapters]" in out


def test_run_baseline_schedule_absent_release_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Valid config with edit.name present
    cfg_path = _write_yaml_cfg(
        tmp_path / "cfg.yaml",
        """
        model:
          id: gpt2
          adapter: hf_gpt2
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
    # Baseline without evaluation_windows should trigger release-mode abort
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"meta": {}}), encoding="utf-8")

    # Stub model_profile to avoid transformers dependency
    run_mod = _import_run_module()
    monkeypatch.setattr(
        run_mod, "detect_model_profile", lambda model_id, adapter: _StubProfile()
    )

    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(
            config=str(cfg_path),
            device="cpu",
            profile="release",
            baseline=str(baseline),
        )
    assert ei.value.exit_code == 1


def test_run_baseline_schedule_mismatch_release_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Valid config with edit.name present
    cfg_path = _write_yaml_cfg(
        tmp_path / "cfg.yaml",
        """
        model:
          id: gpt2
          adapter: hf_gpt2
        edit:
          name: quant_rtn
        dataset:
          provider: wikitext2
          seq_len: 8
          stride: 8
          preview_n: 1
          final_n: 1
        guards:
          order: []
        output:
          dir: runs
        """,
    )
    # Baseline with evaluation_windows present but mismatched stride in data
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "evaluation_windows": {
                    "preview": {"input_ids": [[0, 1, 2]], "window_ids": [1]},
                    "final": {"input_ids": [[3, 4, 5]], "window_ids": [2]},
                },
                "data": {
                    "seq_len": 8,
                    "stride": 4,
                    "dataset": "wikitext2",
                    "split": "validation",
                },
            }
        ),
        encoding="utf-8",
    )

    run_mod = _import_run_module()
    monkeypatch.setattr(
        run_mod, "detect_model_profile", lambda model_id, adapter: _StubProfile()
    )

    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(
            config=str(cfg_path),
            device="cpu",
            profile="release",
            baseline=str(baseline),
        )
    assert ei.value.exit_code == 1


def test_run_device_validation_error_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _write_yaml_cfg(
        tmp_path / "cfg.yaml",
        """
        model:
          id: gpt2
          adapter: hf_gpt2
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
    # Stub profile to avoid heavy imports
    monkeypatch.setattr(
        run_mod, "detect_model_profile", lambda model_id, adapter: _StubProfile()
    )
    # Force device validation failure regardless of resolved device
    import invarlock.cli.device as dev_mod

    monkeypatch.setattr(
        dev_mod, "validate_device_for_config", lambda *a, **k: (False, "bad device")
    )
    with pytest.raises(typer.Exit) as ei:
        run_mod.run_command(
            config=str(cfg_path), device="cpu", profile=None, baseline=None
        )
    assert ei.value.exit_code == 1
