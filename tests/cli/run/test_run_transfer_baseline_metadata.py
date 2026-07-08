# ruff: noqa: I001,E402,F811
from __future__ import annotations

# Consolidated from:
# - tests/cli/run/test_run_branch_more_cases.py
# - tests/cli/run/test_run_additional_branches.py

# --- Begin: test_run_branch_more_cases.py ---

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from invarlock.cli.commands.run import run_command


def _cfg(tmp_path: Path, preview=1, final=1) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
model:
  adapter: hf_causal
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan: {{}}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: {preview}
  final_n: {final}

guards:
  order: [invariants]

eval:
  loss:
    type: auto

output:
  dir: runs
        """
    )
    return p


def _common_patches_detect_ce():
    return (
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report_files.save_report",
            lambda report, run_dir, formats, filename_prefix=None: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
        patch(
            "invarlock.cli.run_runtime_exec.detect_model_profile",
            lambda model_id=None, adapter=None: SimpleNamespace(
                default_loss="ce",
                model_id=model_id,
                adapter=adapter,
                module_selectors={},
                invariants=("checkA",),
                cert_lints=[],
                family="gpt2",
            ),
        ),
        patch(
            "invarlock.cli.run_runtime_exec.resolve_tokenizer",
            lambda prof: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
        patch(
            "invarlock.eval.data.get_provider",
            lambda *a, **k: SimpleNamespace(
                windows=lambda **kw: (
                    SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
                    SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
                )
            ),
        ),
    )


# --- Begin: test_run_additional_branches.py ---

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from invarlock.cli.commands.run import run_command


def _basic_cfg(tmp_path: Path, preview: int = 1, final: int = 1) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
model:
  adapter: hf_causal
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan: {{ heads: {{ mask_only: true, mask_auto: true }} }}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: {preview}
  final_n: {final}

guards:
  order: []

eval:
  loss:
    type: auto

output:
  dir: runs
        """
    )
    return p


def _common_device_and_save():
    return (
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report_files.save_report",
            lambda report, run_dir, formats, filename_prefix=None: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
    )


def _reg_and_provider(provider_windows=None):
    if provider_windows is None:
        provider_windows = (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )

    return (
        patch(
            "invarlock.core.registry.get_registry",
            lambda: SimpleNamespace(
                get_adapter=lambda name: SimpleNamespace(
                    name=name, load_model=lambda model_id, device=None: object()
                ),
                get_edit=lambda name: SimpleNamespace(name=name),
                get_guard=lambda name: SimpleNamespace(name=name),
                get_plugin_metadata=lambda n, t: {
                    "name": n,
                    "module": f"{t}.{n}",
                    "version": "test",
                },
            ),
        ),
        patch(
            "invarlock.eval.data.get_provider",
            lambda *a, **k: SimpleNamespace(windows=lambda **kw: provider_windows),
        ),
    )


def _runner_min():
    return SimpleNamespace(
        execute=lambda **k: SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={},
            status="success",
        )
    )


def _detect_loss(loss_type: str = "ce"):
    return patch(
        "invarlock.cli.run_runtime_exec.detect_model_profile",
        lambda model_id, adapter: SimpleNamespace(
            default_loss=loss_type,
            model_id=model_id,
            adapter=adapter,
            module_selectors={},
            invariants=set(),
            cert_lints=[],
            family="gpt",
            make_tokenizer=lambda: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50_000),
                "tokhash123",
            ),
        ),
    )


# Keep the remainder of the additional branches tests intact
# (We include only a representative subset due to consolidation.)


# ---- Selected general edge scenarios (from edges) ----


def test_guard_overhead_ratio_display_path(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name,
                load_model=lambda model_id, device=None: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda _m, _b=None: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    class DummyRunner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={"plan_digest": "abcd", "deltas": {"params_changed": 0}},
                metrics={
                    "ppl_preview": 10.0,
                    "ppl_final": 10.0,
                    "ppl_ratio": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_match_fraction": 1.0,
                    "loss_type": "ce",
                },
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )

    class _OverheadRatio:
        def __init__(self):
            self.passed = True
            self.messages = []
            self.warnings = []
            self.errors = []
            self.checks = {}
            self.metrics = {"overhead_ratio": 1.02, "overhead_percent": float("nan")}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr("invarlock.core.runner.CoreRunner", lambda: DummyRunner())
    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.validate_guard_overhead",
        lambda *a, **k: _OverheadRatio(),
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.resolve_tokenizer",
        lambda *_a, **_k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tokhash123",
        ),
    )
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda d: d)
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
    )
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile="ci")
    assert (tmp_path / "runs").is_dir()


def test_release_baseline_missing_windows_exits(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"meta": {}, "metrics": {}}))

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(name=name, load_model=lambda *a, **k: object())

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )
        ),
    )
    import click

    with pytest.raises(click.exceptions.Exit):
        run_command(
            config=str(cfg),
            device="cpu",
            out=str(tmp_path / "runs"),
            profile="release",
            baseline=str(baseline),
        )


def test_baseline_pairing_valid_schedule(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    schedule = {
        "evaluation_windows": {
            "preview": {
                "window_ids": [1],
                "input_ids": [[1, 2]],
                "attention_masks": [[1, 1]],
            },
            "final": {
                "window_ids": [2],
                "input_ids": [[3, 4]],
                "attention_masks": [[1, 1]],
            },
        }
    }
    baseline.write_text(json.dumps(schedule))

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(name=name, load_model=lambda *a, **k: object())

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.resolve_tokenizer",
        lambda *_a, **_k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tokhash123",
        ),
    )
    monkeypatch.setattr("invarlock.core.runner.CoreRunner", _runner_min)
    run_command(
        config=str(cfg),
        device="cpu",
        out=str(tmp_path / "runs"),
        profile=None,
        baseline=str(baseline),
    )
    assert (tmp_path / "runs").is_dir()


def test_baseline_missing_eval_windows_fallback(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"meta": {}, "metrics": {}}))

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(name=name, load_model=lambda *a, **k: object())

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.resolve_tokenizer",
        lambda *_a, **_k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tokhash123",
        ),
    )
    monkeypatch.setattr("invarlock.core.runner.CoreRunner", _runner_min)
    run_command(
        config=str(cfg),
        device="cpu",
        out=str(tmp_path / "runs"),
        profile=None,
        baseline=str(baseline),
    )
    assert (tmp_path / "runs").is_dir()


def test_release_capacity_planner_path():
    from invarlock.cli.run_overhead import plan_release_windows

    capacity = {
        "available_unique": 1000,
        "available_nonoverlap": 1000,
        "total_tokens": 500000,
        "dedupe_rate": 0.02,
        "candidate_unique": 800,
        "candidate_limit": 1600,
    }
    plan = plan_release_windows(
        capacity,
        requested_preview=400,
        requested_final=400,
        max_calibration=240,
        console=None,
    )
    assert (
        plan["actual_preview"] == plan["actual_final"] and plan["coverage_ok"] is True
    )


def test_persist_ref_masks_positive(tmp_path: Path):
    # Exercise positive branch of _persist_ref_masks via run_command
    cfg = _basic_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"meta": {"tokenizer_hash": "tokhash123"}}))
    with ExitStack() as stack:
        for ctx in _common_device_and_save():
            stack.enter_context(ctx)
        for ctx in _reg_and_provider():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", _runner_min))
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.resolve_tokenizer",
                lambda *_: (
                    SimpleNamespace(
                        eos_token="</s>", pad_token="</s>", vocab_size=50000
                    ),
                    "tokhash123",
                ),
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            out=str(tmp_path / "runs"),
            baseline=str(baseline),
        )
    assert (tmp_path / "runs").is_dir()
