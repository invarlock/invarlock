from __future__ import annotations

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
from tests.cli.run._support_run_common import configure_guard_metric_impact_skip
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


def _common_patches_mlm():
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
            "invarlock.cli.run_runtime_exec.detect_model_profile",
            lambda model_id, adapter: SimpleNamespace(
                default_loss="mlm",
                default_provider=None,
                default_metric=None,
                model_id=model_id,
                adapter=adapter,
                family="bert",
                module_selectors={},
                invariants=[],
                cert_lints=[],
            ),
        ),
        patch(
            "invarlock.cli.run_runtime_exec.resolve_tokenizer",
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
            "invarlock.reporting.report_bundle.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (filename_prefix + ".json"))
            },
        ),
    )


# --------------------
# Merged from test_run_branch_supplemental.py
# --------------------


def test_baseline_pairing_computes_hashes_and_tokens_in_dataset_meta(tmp_path: Path):
    cfg = configure_guard_metric_impact_skip(_write_base_cfg(tmp_path))
    baseline = tmp_path / "baseline.json"
    preview_ids = [[1, 2, 3], [4, 5]]
    final_ids = [[7, 8], [9, 10, 11]]
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {"window_ids": [0, 1], "input_ids": preview_ids},
                    "final": {"window_ids": [2, 3], "input_ids": final_ids},
                },
            }
        )
    )

    class Provider:
        def windows(self, **kwargs):  # not used since baseline has schedule
            return (
                SimpleNamespace(input_ids=[[1]], attention_masks=[[1]]),
                SimpleNamespace(input_ids=[[2]], attention_masks=[[1]]),
            )

    captured = {}

    def cap_save(report, run_dir, formats, filename_prefix):  # noqa: D401
        captured["report"] = report
        return {"json": str(run_dir / (filename_prefix + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)

        # runner must provide evaluation_windows in release+baseline flow
        def _runner():
            def _exec(**kwargs):
                run_cfg = kwargs.get("config")
                ctx = getattr(run_cfg, "context", {}) if run_cfg is not None else {}
                return SimpleNamespace(
                    edit={},
                    metrics={
                        "primary_metric": {
                            "kind": "ppl_causal",
                            "preview": 1.0,
                            "final": 1.0,
                        },
                        "ppl_preview": 1.0,
                        "ppl_final": 1.0,
                        "ppl_ratio": 1.0,
                        "window_overlap_fraction": 0.0,
                        "window_match_fraction": 1.0,
                        "paired_windows": 2,
                    },
                    guards={},
                    context=ctx,
                    evaluation_windows={
                        "preview": {
                            "window_ids": [0, 1],
                            "input_ids": preview_ids,
                            "attention_masks": [[1] * len(x) for x in preview_ids],
                        },
                        "final": {
                            "window_ids": [2, 3],
                            "input_ids": final_ids,
                            "attention_masks": [[1] * len(x) for x in final_ids],
                        },
                    },
                    status="success",
                )

            return SimpleNamespace(execute=_exec)

        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", _runner))
        stack.enter_context(
            patch("invarlock.reporting.report_bundle.save_report", cap_save)
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
            baseline=str(baseline),
            until_pass=False,
        )

    r = captured["report"]
    assert r["data"]["preview_total_tokens"] == sum(len(x) for x in preview_ids)
    assert r["data"]["final_total_tokens"] == sum(len(x) for x in final_ids)
    assert r["data"]["preview_hash"] == _compute_seq_hash(preview_ids)
    assert r["data"]["final_hash"] == _compute_seq_hash(final_ids)


def test_window_match_fraction_mismatch_exit(tmp_path: Path):
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
                    "window_match_fraction": 0.5,
                },
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )

    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
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


def test_baseline_pairing_propagates_window_plan_capacity(tmp_path: Path):
    cfg = configure_guard_metric_impact_skip(_write_base_cfg(tmp_path))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "data": {
                    "window_plan": {
                        "profile": "release",
                        "requested_preview": 2,
                        "requested_final": 2,
                        "actual_preview": 2,
                        "actual_final": 2,
                        "coverage_ok": True,
                        "preview_total_tokens": 2,
                        "final_total_tokens": 2,
                        "min_tokens_target": 50000,
                        "tokens_floor_met": False,
                        "capacity": {"available_unique": 123},
                    }
                },
                "evaluation_windows": {
                    "preview": {"window_ids": [0, 1], "input_ids": [[1], [2]]},
                    "final": {"window_ids": [2, 3], "input_ids": [[3], [4]]},
                },
            }
        )
    )

    captured = {}

    def cap_save(report, run_dir, formats, filename_prefix):  # noqa: D401
        captured["report"] = report
        return {"json": str(run_dir / (filename_prefix + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)

        # ensure evaluation_windows exist for release+baseline
        def _runner():
            def _exec(**kwargs):
                run_cfg = kwargs.get("config")
                ctx = getattr(run_cfg, "context", {}) if run_cfg is not None else {}
                return SimpleNamespace(
                    edit={},
                    metrics={
                        "primary_metric": {
                            "kind": "ppl_causal",
                            "preview": 1.0,
                            "final": 1.0,
                        },
                        "ppl_preview": 1.0,
                        "ppl_final": 1.0,
                        "ppl_ratio": 1.0,
                        "window_overlap_fraction": 0.0,
                        "window_match_fraction": 1.0,
                        "paired_windows": 2,
                    },
                    guards={},
                    context=ctx,
                    evaluation_windows={
                        "preview": {
                            "window_ids": [0, 1],
                            "input_ids": [[1], [2]],
                            "attention_masks": [[1], [1]],
                        },
                        "final": {
                            "window_ids": [2, 3],
                            "input_ids": [[3], [4]],
                            "attention_masks": [[1], [1]],
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
                        SimpleNamespace(input_ids=[[1]], attention_masks=[[1]]),
                        SimpleNamespace(input_ids=[[2]], attention_masks=[[1]]),
                    )
                ),
            )
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", _runner))
        stack.enter_context(
            patch("invarlock.reporting.report_bundle.save_report", cap_save)
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
            baseline=str(baseline),
            until_pass=False,
        )

    r = captured["report"]
    wc = r["data"].get("window_capacity", {})
    assert wc.get("available_unique") == 123
    assert r["data"]["window_plan"]["preview_total_tokens"] == 2
    assert r["data"]["window_plan"]["final_total_tokens"] == 2
    assert r["data"]["window_plan"]["min_tokens_target"] == 50000
    assert r["data"]["window_plan"]["tokens_floor_met"] is False


def test_invalid_edit_name_triggers_exit(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  adapter: hf_causal
  id: gpt2
  device: cpu
edit:
  name: ""  # invalid
  plan: {}

dataset:
  provider: synthetic
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
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
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
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={
                            "primary_metric": {
                                "kind": "ppl_causal",
                                "preview": 1.0,
                                "final": 1.0,
                            },
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
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile=None,
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_provider_kwargs_propagated(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("ok: true")

    captured = {}

    def fake_get_provider(provider, **kwargs):  # noqa: D401
        captured.update(kwargs)
        return SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )
        )

    with ExitStack() as stack:
        # Patch load_config to include extra dataset kwargs
        class DummyCfg:
            def __init__(self, outdir):
                self.model = SimpleNamespace(
                    id="gpt2", adapter="hf_causal", device="cpu"
                )
                self.edit = SimpleNamespace(name="quant_rtn", plan={})
                self.auto = SimpleNamespace(
                    enabled=False, tier="balanced", probes=0, target_pm_ratio=None
                )
                self.guards = SimpleNamespace(order=[])
                self.dataset = SimpleNamespace(
                    provider="custom",
                    dataset_name="myset",
                    config_name="confA",
                    text_field="content",
                    cache_dir="cache",
                    max_samples=123,
                    split="validation",
                    seq_len=8,
                    stride=4,
                    preview_n=1,
                    final_n=1,
                    seed=42,
                )
                self.eval = SimpleNamespace(
                    spike_threshold=2.0, loss=SimpleNamespace(type="auto")
                )
                self.output = SimpleNamespace(dir=outdir)

            def model_dump(self):
                return {}

        stack.enter_context(
            patch(
                "invarlock.core.config_loader.load_config",
                lambda p: DummyCfg(tmp_path / "runs"),
            )
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", fake_get_provider)
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
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert captured.get("dataset_name") == "myset"
    assert captured.get("config_name") == "confA"
    assert captured.get("text_field") == "content"
    assert captured.get("cache_dir") == "cache"
    assert captured.get("max_samples") == 123


def test_module_selectors_injected_into_edit_config(tmp_path: Path):
    cfg = _write_base_cfg(tmp_path)

    captured = {}

    def runner_factory():
        class R:
            def execute(self, **kwargs):
                captured["edit_config"] = kwargs.get("edit_config")
                return SimpleNamespace(
                    edit={},
                    metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                    guards={},
                    context={"dataset_meta": {}},
                )

        return R()

    with ExitStack() as stack:
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", runner_factory))
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
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
        # override detect_model_profile after common patches
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.detect_model_profile",
                lambda model_id, adapter: SimpleNamespace(
                    default_loss="ce",
                    model_id=model_id,
                    adapter=adapter,
                    module_selectors={"attn": ["q_proj", "k_proj"]},
                    invariants=set(),
                    cert_lints=[],
                    family="gpt2",
                    default_provider=None,
                    default_metric=None,
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

    assert "module_selectors" in captured.get("edit_config", {})
    assert captured["edit_config"]["module_selectors"]  # non-empty mapping injected
