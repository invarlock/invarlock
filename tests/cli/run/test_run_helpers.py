# ruff: noqa: I001,E402,F811
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from invarlock.cli.commands.run import (
    _persist_ref_masks,
    _plan_release_windows,
    run_command,
)
from rich.console import Console


# --------------------
# Release window planner
# --------------------


def test_plan_release_windows_insufficient_capacity():
    capacity = {
        "available_unique": 10,
        "available_nonoverlap": 10,
        "total_tokens": 1000,
        "dedupe_rate": 0.1,
    }
    with pytest.raises(RuntimeError):
        _plan_release_windows(
            capacity, requested_preview=100, requested_final=100, max_calibration=0
        )


def test_plan_release_windows_success_path():
    capacity = {
        "available_unique": 10000,
        "available_nonoverlap": 10000,
        "total_tokens": 10_000_000,
        "dedupe_rate": 0.1,
    }
    plan = _plan_release_windows(
        capacity, requested_preview=500, requested_final=500, max_calibration=100
    )
    assert plan["coverage_ok"] is True
    assert plan["actual_preview"] == plan["actual_final"]
    assert plan["capacity"]["reserve_windows"] >= plan["capacity"]["calibration"]


def test_plan_release_windows_success_and_insufficient(tmp_path: Path):
    # Sufficient capacity
    plan = _plan_release_windows(
        {
            "available_unique": 1000,
            "available_nonoverlap": 1000,
            "total_tokens": 1_000_000,
            "dedupe_rate": 0.1,
        },
        requested_preview=300,
        requested_final=400,
        max_calibration=100,
        console=None,
    )
    assert (
        plan["coverage_ok"] is True
        and plan["actual_preview"] > 0
        and plan["capacity"]["reserve_windows"] > 0
    )

    # Insufficient capacity triggers RuntimeError
    with pytest.raises(RuntimeError):
        _ = _plan_release_windows(
            {
                "available_unique": 10,
                "available_nonoverlap": 10,
                "total_tokens": 1000,
                "dedupe_rate": 0.1,
            },
            requested_preview=10,
            requested_final=10,
            max_calibration=0,
            console=None,
        )

    # Candidate unique path
    plan2 = _plan_release_windows(
        {
            "available_unique": 1000,
            "available_nonoverlap": 1000,
            "total_tokens": 1_000_000,
            "dedupe_rate": 0.1,
            "candidate_unique": 800,
            "candidate_limit": 1200,
        },
        requested_preview=200,
        requested_final=200,
        max_calibration=50,
        console=None,
    )
    assert plan2["capacity"]["candidate_unique"] == 800


# --------------------
# Persist edit mask artifacts
# --------------------


def test_persist_ref_masks(tmp_path: Path):
    core_report = SimpleNamespace(
        edit={"artifacts": {"mask_payload": {"indices": [1, 2, 3], "meta": {}}}}
    )
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    mask_path = _persist_ref_masks(core_report, out_dir)
    assert mask_path and mask_path.exists()


# --------------------
# Baseline pairing: attention masks generation from windows
# --------------------


def test_run_command_baseline_pairing_generates_attention_masks(tmp_path: Path):
    # Config
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
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    # No attention_masks provided to exercise fallback generation
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 2, 3, 4]],
                        "labels": [[-100, -100, 3, -100]],
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[5, 6, 7, 8]],
                        "labels": [[-100, -100, -100, 8]],
                    },
                },
            }
        )
    )

    class DummyRegistry:
        def get_adapter(self, name):  # noqa: ARG002 - stub
            return SimpleNamespace(
                name=name, load_model=lambda model_id, device: object()
            )

        def get_edit(self, name):  # noqa: ARG002 - stub
            return SimpleNamespace(name=name)

        def get_guard(self, name):  # noqa: ARG002 - stub
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, t):  # noqa: ARG002 - stub
            return {"name": name, "module": f"{t}.{name}", "version": "test"}

    class DummyRunner:
        def execute(self, **kwargs):
            # Pass through the baseline windows to ensure they are used
            eval_windows = {
                "preview": {
                    "window_ids": [0],
                    "logloss": [3.0],
                    "input_ids": [[1, 2, 3, 4]],
                    "token_counts": [4],
                    "masked_token_counts": [1],
                    "actual_token_counts": [4],
                    "labels": [[-100, -100, 3, -100]],
                },
                "final": {
                    "window_ids": [1],
                    "logloss": [3.2],
                    "input_ids": [[5, 6, 7, 8]],
                    "token_counts": [4],
                    "masked_token_counts": [1],
                    "actual_token_counts": [4],
                    "labels": [[-100, -100, -100, 8]],
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
                    "paired_windows": 1,
                    "loss_type": "mlm",
                },
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows=eval_windows,
                status="success",
            )

    outdir = tmp_path / "runs"
    with (
        patch("invarlock.core.registry.get_registry", lambda: DummyRegistry()),
        patch("invarlock.core.runner.CoreRunner", lambda: DummyRunner()),
        patch(
            "invarlock.eval.data.get_provider",
            lambda *args, **kwargs: SimpleNamespace(
                windows=lambda **k: (
                    SimpleNamespace(
                        input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]]
                    ),
                    SimpleNamespace(
                        input_ids=[[5, 6, 7, 8]], attention_masks=[[1, 1, 1, 1]]
                    ),
                )
            ),
        ),
        patch(
            "invarlock.cli.commands.run.detect_model_profile",
            lambda model_id, adapter: SimpleNamespace(
                default_loss="mlm",
                model_id=model_id,
                adapter=adapter,
                module_selectors={},
                invariants=[],
                cert_lints=[],
                family="bert",
                make_tokenizer=lambda: (
                    SimpleNamespace(
                        mask_token_id=103,
                        eos_token="</s>",
                        pad_token="</s>",
                        vocab_size=50000,
                        all_special_ids=[0, 1, 2],
                    ),
                    "tokhash123",
                ),
            ),
        ),
        patch(
            "invarlock.cli.commands.run.resolve_tokenizer",
            lambda model_profile: (
                SimpleNamespace(
                    mask_token_id=103,
                    eos_token="</s>",
                    pad_token="</s>",
                    vocab_size=50000,
                    all_special_ids=[0, 1, 2],
                ),
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
        # Should succeed and implicitly generate attention masks from input_ids
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            out=str(outdir),
            baseline=str(baseline),
            until_pass=False,
        )


# --------------------
# Serialization helper path: prefers .dict() if present
# --------------------


def _cfg_path(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
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
  preview_n: 1
  final_n: 1

guards:
  order: []

eval:
  spike_threshold: 2.0
  loss:
    type: auto

output:
  dir: runs
        """
    )
    return p


def test_to_serialisable_dict_uses_dict_method(tmp_path: Path):
    cfg_path = _cfg_path(tmp_path)

    class EvalObj:
        def __init__(self):
            self.spike_threshold = 2.0
            self.loss = SimpleNamespace(type="auto")

        def dict(self):
            return {
                "spike_threshold": self.spike_threshold,
                "loss": {"type": self.loss.type},
            }

    class Cfg:
        def __init__(self):
            self.model = SimpleNamespace(adapter="hf_gpt2", id="gpt2", device="cpu")
            self.edit = SimpleNamespace(name="quant_rtn", plan={})
            self.dataset = SimpleNamespace(
                provider="synthetic",
                id="synthetic",
                split="validation",
                seq_len=8,
                stride=4,
                preview_n=1,
                final_n=1,
                seed=42,
            )
            self.guards = SimpleNamespace(order=[])
            self.eval = EvalObj()
            self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
            self.output = SimpleNamespace(dir=tmp_path / "runs")

        def model_dump(self):
            return {}

    with ExitStack() as stack:
        stack.enter_context(patch("invarlock.cli.config.load_config", lambda p: Cfg()))
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]
                        ),
                    )
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                        },
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        stack.enter_context(patch("invarlock.cli.device.resolve_device", lambda d: d))
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
        run_command(config=str(cfg_path), device="cpu", out=str(tmp_path / "runs"))


# --------------------
# Coverage boosters: negative paths for helpers
# --------------------


def test_persist_masks_returns_none_when_no_edit_dict(tmp_path: Path):
    core_report = SimpleNamespace(edit=None)
    assert _persist_ref_masks(core_report, tmp_path) is None


def test_persist_masks_returns_none_when_no_artifacts_dict(tmp_path: Path):
    core_report = SimpleNamespace(edit={"artifacts": None})
    assert _persist_ref_masks(core_report, tmp_path) is None


def test_persist_masks_returns_none_when_mask_payload_invalid(tmp_path: Path):
    core_report = SimpleNamespace(edit={"artifacts": {"mask_payload": []}})
    assert _persist_ref_masks(core_report, tmp_path) is None


def test_plan_release_windows_console_adjustment_branch():
    capacity = {
        "available_unique": 2000,
        "available_nonoverlap": 2000,
        "total_tokens": 1_000_000,
        "dedupe_rate": 0.1,
        "candidate_unique": 800,
        "candidate_limit": 1200,
    }
    plan = _plan_release_windows(
        capacity,
        requested_preview=500,
        requested_final=500,
        max_calibration=100,
        console=Console(),
    )
    assert plan["coverage_ok"] is True
    assert plan["actual_preview"] < plan["target_per_arm"]
    assert plan["capacity"].get("candidate_unique") == 800
    assert plan["capacity"].get("candidate_limit") == 1200
