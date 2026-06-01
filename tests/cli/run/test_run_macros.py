from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import typer

from tests.cli._support_transformers import install_transformers_tokenizer_stub


def _import_run_module():
    install_transformers_tokenizer_stub()

    import importlib

    return importlib.import_module("invarlock.cli.commands.run")


class _EvalWin:
    def __init__(self, ids):
        self.input_ids = ids
        # simple labels with -100 and valid positions
        self.labels = [[-100 if i % 2 == 0 else 1 for i in row] for row in ids]
        self.attention_masks = [[1 for _ in row] for row in ids]


class _ProviderDev:
    def available_splits(self):
        return ["validation"]

    def windows(self, *, tokenizer, seq_len, stride, preview_n, final_n, seed, split):
        prev = _EvalWin([[1] * seq_len for _ in range(preview_n)])
        fin = _EvalWin([[1] * seq_len for _ in range(final_n)])
        return prev, fin


class _ProviderRelease(_ProviderDev):
    def estimate_capacity(self, **kwargs):  # noqa: D401
        # Minimal capacity sufficient for plan
        return {
            "available_unique": 1000,
            "available_nonoverlap": 1000,
            "total_tokens": 100000,
            "dedupe_rate": 0.1,
        }


class _Adapter:
    name = "fake_adapter"

    def load_model(self, *a, **k):
        return object()


class _Registry:
    def get_adapter(self, name):
        return _Adapter()

    def get_edit(self, name):
        return types.SimpleNamespace(name=name)

    def get_plugin_metadata(self, name, typ):  # used for provenance
        return {"name": name, "type": typ}


class _CoreReport:
    def __init__(self):
        self.context = {
            "window_plan": {
                "profile": "release",
                "requested_preview": 1,
                "requested_final": 1,
                "actual_preview": 1,
                "actual_final": 1,
                "capacity": {"available_unique": 100, "reserve_windows": 10},
            }
        }
        self.edit = {"deltas": {"params_changed": 0}}
        self.metrics = {"latency_ms_per_tok": 0.0, "memory_mb_peak": 0.0}
        self.guards = {}
        self.evaluation_windows = {
            "preview": {
                "input_ids": [[1, 1, 1, 1]],
                "labels": [[-100, 1, -100, 1]],
                "window_ids": [1],
            },
            "final": {
                "input_ids": [[1, 1, 1, 1]],
                "labels": [[-100, 1, -100, 1]],
                "window_ids": [2],
            },
        }


class _CoreRunnerExit:
    def __init__(self):
        pass

    def execute(self, **kwargs):
        return _CoreReport()


def _patch_common(monkeypatch, run_mod, provider):
    # Registry and provider (patch source modules to affect inside-function imports)
    import invarlock.reporting.report_files as report_files_mod

    import invarlock.cli.device as dev_mod
    import invarlock.cli.run_runtime_exec as runtime_mod
    import invarlock.core.registry as reg_mod
    import invarlock.core.runner as runner_mod
    import invarlock.eval.data as data_mod

    monkeypatch.setattr(reg_mod, "get_registry", lambda: _Registry())
    monkeypatch.setattr(data_mod, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(data_mod, "EvaluationWindow", _EvalWin)
    # Tokenizer resolve
    monkeypatch.setattr(
        runtime_mod,
        "resolve_tokenizer",
        lambda *a, **k: (
            types.SimpleNamespace(
                eos_token="</s>",
                pad_token="</s>",
                eos_token_id=1,
                pad_token_id=0,
                vocab_size=32,
                get_vocab=lambda: {"<pad>": 0, "</s>": 1},
            ),
            "tok",
        ),
    )
    # Device is valid CPU
    monkeypatch.setattr(dev_mod, "resolve_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(
        dev_mod, "validate_device_for_config", lambda *a, **k: (True, "")
    )
    # Provide lightweight CoreRunner implementation
    monkeypatch.setattr(runner_mod, "CoreRunner", _CoreRunnerExit)

    # Save report stub that keeps the macro path lightweight but successful.
    def _save_report_stub(report, out_dir, formats=None, filename_prefix="report"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"{filename_prefix}.json"
        report_path.write_text(json.dumps(report, default=str), encoding="utf-8")
        return {"json": report_path}

    monkeypatch.setattr(report_files_mod, "save_report", _save_report_stub)


def _write_cfg(path: Path, content: dict) -> Path:
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def _write_baseline(path: Path, preview: int = 1, final: int = 1) -> Path:
    payload = {
        "evaluation_windows": {
            "preview": {
                "input_ids": [[0] * 4 for _ in range(preview)],
                "window_ids": list(range(preview)),
            },
            "final": {
                "input_ids": [[1] * 4 for _ in range(final)],
                "window_ids": list(range(preview, preview + final)),
            },
        },
        "data": {
            "seq_len": 4,
            "stride": 4,
            "dataset": "wikitext2",
            "split": "validation",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_macro_dev_flow_quick_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_mod = _import_run_module()
    cfg = _write_cfg(
        tmp_path / "cfg.yaml",
        {
            "model": {"id": "gpt2", "adapter": "hf_causal"},
            "edit": {"name": "quant_rtn"},
            "dataset": {"provider": "wikitext2", "seq_len": 4, "stride": 4},
            "eval": {},
            "guards": {"order": []},
            "output": {"dir": "runs"},
        },
    )
    _patch_common(monkeypatch, run_mod, _ProviderDev())
    report_path = run_mod.run_command(
        config=str(cfg), device="cpu", profile="dev", baseline=None
    )
    assert isinstance(report_path, Path)
    assert report_path.suffix == ".json"


def test_run_macro_release_flow_with_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_mod = _import_run_module()
    cfg = _write_cfg(
        tmp_path / "cfg.yaml",
        {
            "model": {"id": "gpt2", "adapter": "hf_causal"},
            "edit": {"name": "quant_rtn"},
            "dataset": {
                "provider": "wikitext2",
                "seq_len": 4,
                "stride": 4,
                "preview_n": 1,
                "final_n": 1,
            },
            "eval": {},
            "guards": {"order": [], "variance": {"max_calib": 100}},
            "output": {"dir": "runs"},
        },
    )
    _patch_common(monkeypatch, run_mod, _ProviderRelease())
    try:
        report_path = run_mod.run_command(
            config=str(cfg), device="cpu", profile="release", baseline=None
        )
    except typer.Exit as exc:
        assert exc.exit_code == 1
    else:
        assert isinstance(report_path, Path)
        assert report_path.suffix == ".json"


def test_run_macro_with_baseline_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_mod = _import_run_module()
    cfg = _write_cfg(
        tmp_path / "cfg.yaml",
        {
            "model": {"id": "gpt2", "adapter": "hf_causal"},
            "edit": {"name": "quant_rtn"},
            "dataset": {"provider": "wikitext2", "seq_len": 4, "stride": 4},
            "eval": {},
            "guards": {"order": []},
            "output": {"dir": "runs"},
        },
    )
    baseline = _write_baseline(tmp_path / "baseline.json", preview=1, final=1)
    _patch_common(monkeypatch, run_mod, _ProviderDev())
    report_path = run_mod.run_command(
        config=str(cfg), device="cpu", profile="dev", baseline=str(baseline)
    )
    assert isinstance(report_path, Path)
    assert report_path.suffix == ".json"
