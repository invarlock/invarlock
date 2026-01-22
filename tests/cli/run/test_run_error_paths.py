import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command


def _write_min_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  adapter: hf_causal
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


def _common_patches():
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
            "invarlock.eval.data.get_provider",
            lambda *a, **k: SimpleNamespace(
                estimate_capacity=lambda **kw: {
                    "available_unique": 2000,
                    "available_nonoverlap": 2000,
                    "total_tokens": 1000000,
                    "dedupe_rate": 0.1,
                },
                windows=lambda **kw: (
                    SimpleNamespace(
                        input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]]
                    ),
                    SimpleNamespace(
                        input_ids=[[5, 6, 7, 8]], attention_masks=[[1, 1, 1, 1]]
                    ),
                ),
            ),
        ),
        patch(
            "invarlock.cli.commands.run.detect_model_profile",
            lambda model_id, adapter: SimpleNamespace(
                default_loss="ce", model_id=model_id, adapter=adapter
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
    )


def test_run_command_invalid_tier_and_probes_exit(tmp_path: Path):
    cfg = _write_min_config(tmp_path)
    outdir = tmp_path / "runs"
    with ExitStack() as stack:
        for ctx in _common_patches():
            stack.enter_context(ctx)
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile=None,
                out=str(outdir),
                tier="invalid",
                probes=0,
                until_pass=False,
            )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile=None,
                out=str(outdir),
                tier="balanced",
                probes=99,  # invalid range
                until_pass=False,
            )


def test_run_command_release_missing_eval_windows_in_baseline_exits(tmp_path: Path):
    cfg = _write_min_config(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"meta": {"tokenizer_hash": "tokhash123"}}))
    outdir = tmp_path / "runs"
    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
        for ctx in _common_patches():
            stack.enter_context(ctx)
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            out=str(outdir),
            baseline=str(baseline),
            until_pass=False,
        )


def test_run_command_mlm_pairing_missing_labels_exits(tmp_path: Path):
    cfg = _write_min_config(tmp_path)
    # Baseline with eval windows but without labels/masked counts → should fail in MLM pairing
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123", "window_plan": {}},
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 2, 3, 4]],
                        "attention_masks": [[1, 1, 1, 1]],
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[5, 6, 7, 8]],
                        "attention_masks": [[1, 1, 1, 1]],
                    },
                },
            }
        )
    )
    outdir = tmp_path / "runs"
    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
        for ctx in _common_patches():
            stack.enter_context(ctx)
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            out=str(outdir),
            baseline=str(baseline),
            until_pass=False,
        )


def test_missing_edit_name_exits(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  adapter: hf_causal
  id: gpt2
  device: cpu
edit:
  name: ""
  plan: {}

dataset:
  provider: synthetic
  id: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: 1
  final_n: 1

guards:
  order: []

eval:
  loss:
    type: auto

output:
  dir: runs
        """
    )
    with ExitStack() as stack:
        for ctx in _common_patches():
            stack.enter_context(ctx)
        with pytest.raises(click.exceptions.Exit):
            run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
