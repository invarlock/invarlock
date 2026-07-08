from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command
from tests.cli.run._support_run_common import (
    SNS as _SNS,
)
from tests.cli.run._support_run_pairing import (
    baseline_pairing_common_patches_ce as _common_patches_ce,
)
from tests.cli.run._support_run_pairing import (
    baseline_pairing_compute_seq_hash as _compute_seq_hash,
)
from tests.cli.run._support_run_pairing import (
    baseline_pairing_write_base_cfg as _write_base_cfg,
)
from tests.cli.run._support_run_pairing import (
    supplemental_cfg as _supp_cfg,
)
from tests.cli.run._support_run_pairing import (
    supplemental_common_patches_detect_ce as _supp_common_patches_detect_ce,
)
from tests.cli.run._support_run_pairing import (
    supplemental_provider_min as _supp_provider_min,
)


def test_metrics_window_plan_stats_and_capacity_mapping(tmp_path: Path):
    cfg = _write_base_cfg(tmp_path)

    captured = {}

    def cap_save(report, run_dir, formats, filename_prefix):
        captured["report"] = report
        return {"json": str(run_dir / (filename_prefix + ".json"))}

    def runner_factory():
        class R:
            def execute(self, **kwargs):
                ctx = {
                    "dataset_meta": {},
                    "window_plan": {
                        "requested_preview": 3,
                        "requested_final": 4,
                        "actual_preview": 2,
                        "actual_final": 2,
                        "coverage_ok": True,
                        "preview_total_tokens": 24,
                        "final_total_tokens": 18,
                        "min_tokens_target": 50000,
                        "tokens_floor_met": False,
                        "dedupe_adjustments": [{"deficit": 2, "proposed_per_arm": 2}],
                        "capacity": {"available_unique": 999},
                    },
                }
                return SimpleNamespace(
                    edit={},
                    metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                    guards={},
                    context=ctx,
                )

        return R()

    with ExitStack() as stack:
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.detect_model_profile",
                lambda model_id, adapter: SimpleNamespace(
                    default_loss=None,
                    default_provider=None,
                    default_metric=None,
                    model_id=model_id,
                    adapter=adapter,
                    family="gpt2",
                    module_selectors={},
                    invariants=[],
                    cert_lints=[],
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                        SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
                    )
                ),
            )
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", runner_factory))
        stack.enter_context(
            patch("invarlock.reporting.report_files.save_report", cap_save)
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    metrics = captured["report"]["metrics"]
    assert metrics["stats"]["requested_preview"] == 3
    assert metrics["stats"]["requested_final"] == 4
    assert metrics["stats"]["actual_preview"] == 2
    assert metrics["stats"]["actual_final"] == 2
    assert metrics["stats"]["preview_total_tokens"] == 24
    assert metrics["stats"]["final_total_tokens"] == 18
    assert metrics["stats"]["min_tokens_target"] == 50000
    assert metrics["stats"]["tokens_floor_met"] is False
    assert metrics["stats"]["dedupe_adjustments"] == [
        {"deficit": 2, "proposed_per_arm": 2}
    ]
    assert metrics.get("window_capacity", {}).get("available_unique") == 999


def test_metrics_loss_type_fallback_from_dataset_meta_context(tmp_path: Path):
    cfg = _write_base_cfg(tmp_path)

    captured = {}

    def cap_save(report, run_dir, formats, filename_prefix):
        captured["report"] = report
        return {"json": str(run_dir / (filename_prefix + ".json"))}

    def runner_factory():
        class R:
            def execute(self, **kwargs):
                ctx = {"dataset_meta": {"loss_type": "causal"}}
                return SimpleNamespace(
                    edit={},
                    metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                    guards={},
                    context=ctx,
                )

        return R()

    with ExitStack() as stack:
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(input_ids=[[1]], attention_masks=[[1]]),
                        SimpleNamespace(input_ids=[[2]], attention_masks=[[1]]),
                    )
                ),
            )
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", runner_factory))
        stack.enter_context(
            patch("invarlock.reporting.report_files.save_report", cap_save)
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert captured["report"]["metrics"]["loss_type"] in {"causal", "ce", "mlm"}


def test_device_validation_failure_exits(tmp_path: Path):
    cfg = _write_base_cfg(tmp_path)

    with ExitStack() as stack:
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config",
                lambda d: (False, "unsupported device"),
            )
        )
        with pytest.raises(click.exceptions.Exit) as excinfo:
            run_command(
                config=str(cfg),
                device="cpu",
                profile=None,
                out=str(tmp_path / "runs"),
                until_pass=False,
            )

    assert excinfo.value.exit_code == 1


def test_report_meta_includes_tokenizer_hash_on_provider_path(tmp_path: Path):
    cfg = _write_base_cfg(tmp_path)

    captured = {}

    def cap_save(report, run_dir, formats, filename_prefix):
        captured["report"] = report
        return {"json": str(run_dir / (filename_prefix + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
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
                    )
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.reporting.report_files.save_report", cap_save)
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert isinstance(captured["report"]["meta"].get("tokenizer_hash"), str)


def test_noop_guard_is_ignored(tmp_path: Path):
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
  split: validation
  seq_len: 8
  stride: 4
  preview_n: 1
  final_n: 1

guards:
  order: [noop]

eval:
  loss:
    type: auto

output:
  dir: runs
        """
    )

    captured = {}

    def runner_factory():
        class R:
            def execute(self, **kwargs):
                captured["guards"] = kwargs.get("guards")
                return SimpleNamespace(
                    edit={},
                    metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                    guards={},
                    context={"dataset_meta": {}},
                )

        return R()

    with ExitStack() as stack:
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", runner_factory))
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                        SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
                    )
                ),
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert captured.get("guards") == []


def test_baseline_pairing_respects_existing_hashes_in_meta(tmp_path: Path):
    cfg = _write_base_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    preview_ids = [[1, 2, 3]]
    final_ids = [[4, 5, 6]]
    pre_hash = _compute_seq_hash(preview_ids)
    fin_hash = _compute_seq_hash(final_ids)

    ds_hash = hashlib.blake2s(
        (pre_hash + fin_hash).encode("utf-8"), digest_size=16
    ).hexdigest()
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "data": {
                    "preview_hash": pre_hash,
                    "final_hash": fin_hash,
                    "dataset_hash": ds_hash,
                },
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": preview_ids},
                    "final": {"window_ids": [1], "input_ids": final_ids},
                },
            }
        )
    )

    captured = {}

    def cap_save(report, run_dir, formats, filename_prefix):
        captured["report"] = report
        return {"json": str(run_dir / (filename_prefix + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)

        def _runner():
            def _exec(**kwargs):
                run_cfg = kwargs.get("config")
                ctx = getattr(run_cfg, "context", {}) if run_cfg is not None else {}
                return SimpleNamespace(
                    edit={},
                    metrics={
                        "ppl_preview": 1.0,
                        "ppl_final": 1.0,
                        "ppl_ratio": 1.0,
                        "window_overlap_fraction": 0.0,
                        "window_match_fraction": 1.0,
                        "paired_windows": 1,
                    },
                    guards={},
                    context=ctx,
                    evaluation_windows={
                        "preview": {
                            "window_ids": [0],
                            "input_ids": preview_ids,
                            "attention_masks": [[1, 1, 1]],
                        },
                        "final": {
                            "window_ids": [1],
                            "input_ids": final_ids,
                            "attention_masks": [[1, 1, 1]],
                        },
                    },
                    status="success",
                )

            return SimpleNamespace(execute=_exec)

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
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", _runner))
        stack.enter_context(
            patch("invarlock.reporting.report_files.save_report", cap_save)
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            out=str(tmp_path / "runs"),
            baseline=str(baseline),
            until_pass=False,
        )

    data = captured["report"]["data"]
    assert data["preview_hash"] == pre_hash
    assert data["final_hash"] == fin_hash
    assert data["dataset_hash"] == ds_hash


def test_metrics_inherits_masked_token_counts_from_dataset_meta_context(tmp_path: Path):
    cfg = _write_base_cfg(tmp_path)
    captured = {}

    def cap_save(report, run_dir, formats, filename_prefix):
        captured["report"] = report
        return {"json": str(run_dir / (filename_prefix + ".json"))}

    def runner_factory():
        class R:
            def execute(self, **kwargs):
                ctx = {
                    "dataset_meta": {
                        "masked_tokens_total": 5,
                        "masked_tokens_preview": 2,
                        "masked_tokens_final": 3,
                    }
                }
                return SimpleNamespace(
                    edit={},
                    metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                    guards={},
                    context=ctx,
                )

        return R()

    with ExitStack() as stack:
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
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
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", runner_factory))
        stack.enter_context(
            patch("invarlock.reporting.report_files.save_report", cap_save)
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    metrics = captured["report"]["metrics"]
    assert metrics.get("masked_tokens_total") == 5
    assert metrics.get("masked_tokens_preview") == 2
    assert metrics.get("masked_tokens_final") == 3


def test_dataset_meta_context_non_dict_path(tmp_path: Path):
    cfg = _supp_cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return _SNS(
                edit={},
                metrics={
                    "ppl_preview": 1.0,
                    "ppl_final": 1.0,
                    "ppl_ratio": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_match_fraction": 1.0,
                },
                guards={},
                context={"dataset_meta": [1, 2, 3]},
                evaluation_windows={},
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _supp_common_patches_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _supp_provider_min()
            )
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.resolve_tokenizer",
                lambda prof: (
                    _SNS(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                    "tokhash123",
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
    assert (tmp_path / "runs").is_dir()
