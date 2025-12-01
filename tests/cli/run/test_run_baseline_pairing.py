from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command


def test_run_command_baseline_pairing_ce_without_attention_masks(tmp_path: Path):
    # Config with default_loss=ce path
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  adapter: hf_gpt2
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan: {}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: 2
  final_n: 2

guards:
  order: []

eval:
  loss:
    type: auto

output:
  dir: runs
        """
    )

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123", "window_plan": {}},
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0, 1],
                        "input_ids": [[1, 2, 3], [4, 5, 6]],
                    },
                    "final": {
                        "window_ids": [2, 3],
                        "input_ids": [[7, 8, 9], [10, 11, 12]],
                    },
                },
            }
        )
    )

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name, load_model=lambda model_id, device: object()
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    class DummyRunner:
        def execute(self, **kwargs):
            eval_windows = {
                "preview": {
                    "window_ids": [0, 1],
                    "logloss": [3.0, 3.1],
                    "input_ids": [[1, 2, 3], [4, 5, 6]],
                    "attention_masks": [[1, 1, 1], [1, 1, 1]],
                    "token_counts": [3, 3],
                    "actual_token_counts": [3, 3],
                },
                "final": {
                    "window_ids": [2, 3],
                    "logloss": [3.2, 3.3],
                    "input_ids": [[7, 8, 9], [10, 11, 12]],
                    "attention_masks": [[1, 1, 1], [1, 1, 1]],
                    "token_counts": [3, 3],
                    "actual_token_counts": [3, 3],
                },
            }
            return SimpleNamespace(
                edit={
                    "plan_digest": "abcd",
                    "deltas": {
                        "params_changed": 0,
                        "heads_pruned": 0,
                        "neurons_pruned": 0,
                        "layers_modified": 0,
                    },
                },
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
                evaluation_windows=eval_windows,
                status="success",
            )

    class Provider:
        def windows(self, **kwargs):
            prev = SimpleNamespace(
                input_ids=[[1, 2, 3], [4, 5, 6]], attention_masks=[[1, 1, 1], [1, 1, 1]]
            )
            fin = SimpleNamespace(
                input_ids=[[7, 8, 9], [10, 11, 12]],
                attention_masks=[[1, 1, 1], [1, 1, 1]],
            )
            return prev, fin

    outdir = tmp_path / "runs"
    with (
        patch("invarlock.core.registry.get_registry", lambda: DummyRegistry()),
        patch("invarlock.core.runner.CoreRunner", lambda: DummyRunner()),
        patch("invarlock.eval.data.get_provider", lambda *args, **kwargs: Provider()),
        patch(
            "invarlock.cli.commands.run.detect_model_profile",
            lambda model_id, adapter: SimpleNamespace(
                default_loss="ce",
                default_provider=None,
                default_metric=None,
                model_id=model_id,
                adapter=adapter,
                family="gpt2",
                module_selectors={},
                invariants=[],
                cert_lints=[],
            ),
        ),
        patch(
            "invarlock.cli.commands.run.resolve_tokenizer",
            lambda model_profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (filename_prefix + ".json"))
            },
        ),
        patch(
            "invarlock.cli.commands.run.validate_guard_overhead",
            lambda *args, **kwargs: SimpleNamespace(
                passed=True,
                overhead_ratio=0.0,
                overhead_percent=0.0,
                threshold=0.01,
                errors=[],
            ),
        ),
    ):
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            out=str(outdir),
            edit=None,
            tier=None,
            probes=0,
            until_pass=False,
            baseline=str(baseline),
        )


def _write_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  adapter: hf_gpt2
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan: {}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: 2
  final_n: 2

guards:
  order: []

eval:
  loss:
    type: auto

output:
  dir: runs
        """
    )
    return cfg


def _common_patches_for_baseline():
    return (
        patch(
            "invarlock.core.registry.get_registry",
            lambda: SimpleNamespace(
                get_adapter=lambda name: SimpleNamespace(
                    name=name, load_model=lambda model_id, device: object()
                ),
                get_edit=lambda name: SimpleNamespace(name=name),
                get_guard=lambda name: (_ for _ in ()).throw(KeyError("no guard")),
                get_plugin_metadata=lambda n, t: {
                    "name": n,
                    "module": f"{t}.{n}",
                    "version": "test",
                },
            ),
        ),
        patch(
            "invarlock.core.runner.CoreRunner",
            lambda: SimpleNamespace(
                execute=lambda **k: SimpleNamespace(
                    edit={
                        "plan_digest": "abcd",
                        "deltas": {
                            "params_changed": 0,
                            "heads_pruned": 0,
                            "neurons_pruned": 0,
                            "layers_modified": 0,
                        },
                    },
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
            ),
        ),
        patch(
            "invarlock.cli.commands.run.detect_model_profile",
            lambda model_id, adapter: SimpleNamespace(
                default_loss="ce", model_id=model_id, adapter=adapter
            ),
        ),
        patch(
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (filename_prefix + ".json"))
            },
        ),
        patch(
            "invarlock.cli.commands.run.resolve_tokenizer",
            lambda model_profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
    )


def test_baseline_pairing_seq_len_mismatch_exit(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "data": {
                    "seq_len": 16,
                    "dataset": "synthetic",
                    "split": "validation",
                    "stride": 4,
                },
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [1], "input_ids": [[4, 5, 6]]},
                },
            }
        )
    )
    outdir = tmp_path / "runs"
    with (
        patch(
            "invarlock.eval.data.get_provider",
            lambda *args, **kwargs: SimpleNamespace(
                windows=lambda **k: (
                    SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
                    SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
                )
            ),
        ),
        pytest.raises(click.exceptions.Exit),
    ):
        with ExitStack() as stack:
            for ctx in _common_patches_for_baseline():
                stack.enter_context(ctx)
            run_command(
                config=str(cfg),
                device="cpu",
                profile="release",
                out=str(outdir),
                baseline=str(baseline),
                until_pass=False,
            )


def test_baseline_pairing_dataset_split_mismatch_exit(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "data": {
                    "dataset": "other",
                    "split": "train",
                    "seq_len": 8,
                    "stride": 4,
                },
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [1], "input_ids": [[4, 5, 6]]},
                },
            }
        )
    )
    outdir = tmp_path / "runs"
    with (
        patch(
            "invarlock.eval.data.get_provider",
            lambda *args, **kwargs: SimpleNamespace(
                windows=lambda **k: (
                    SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
                    SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
                )
            ),
        ),
        pytest.raises(click.exceptions.Exit),
    ):
        with ExitStack() as stack:
            for ctx in _common_patches_for_baseline():
                stack.enter_context(ctx)
            run_command(
                config=str(cfg),
                device="cpu",
                profile="release",
                out=str(outdir),
                baseline=str(baseline),
                until_pass=False,
            )


def test_provider_eval_window_count_mismatch_exit(tmp_path: Path):
    from invarlock.eval.data import EvaluationWindow

    cfg = _write_cfg(tmp_path)
    outdir = tmp_path / "runs"

    class Provider:
        def windows(self, **kwargs):
            prev = EvaluationWindow(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]])
            fin = EvaluationWindow(input_ids=[], attention_masks=[])
            return prev, fin

    with (
        patch("invarlock.eval.data.get_provider", lambda *args, **kwargs: Provider()),
        pytest.raises(click.exceptions.Exit),
    ):
        with ExitStack() as stack:
            for ctx in _common_patches_for_baseline():
                stack.enter_context(ctx)
            run_command(
                config=str(cfg),
                device="cpu",
                profile=None,
                out=str(outdir),
                until_pass=False,
            )
