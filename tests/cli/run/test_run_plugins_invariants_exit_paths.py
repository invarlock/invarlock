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
    common_ce_patches,
    write_base_run_config,
)
from tests.cli.run._support_run_common import (
    synthetic_provider_min as _provider_min,
)


def _write_cfg(tmp_path: Path, preview=2, final=2, loss_type="auto") -> Path:
    return write_base_run_config(
        tmp_path,
        preview,
        final,
        eval_fields="  spike_threshold: 2.0\n",
        loss_type=loss_type,
    )


def _common_ce():
    return common_ce_patches(
        include_registry=True,
        include_save_report=True,
    )


def _baseline_with_meta(tmp_path: Path, meta: dict, preview_ids, final_ids) -> Path:
    p = tmp_path / "baseline.json"
    payload = {
        "meta": meta,
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.0,
                "final": 1.0,
            }
        },
        "edit": {
            "name": "structured",
            "plan_digest": "baseline",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {
            "preview": {"window_ids": [0], "input_ids": preview_ids},
            "final": {"window_ids": [1], "input_ids": final_ids},
        },
    }
    p.write_text(json.dumps(payload))
    return p


def test_profile_apply_failure_exit(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )

        def runner_exec(**kwargs):
            cfg_ctx = getattr(kwargs.get("config"), "context", {})
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context=cfg_ctx,
                status="success",
            )

        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.config_loader.apply_profile",
                side_effect=RuntimeError("bad profile"),
            )
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile="ci",
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_edit_override_ok(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.run_config._resolve_requested_edit_name",
                lambda name: name,
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.run_config._apply_requested_edit_override",
                lambda c, e, *, config_cls: c,
            )
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )

        def runner_exec2(**kwargs):
            cfg_ctx = getattr(kwargs.get("config"), "context", {})
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context=cfg_ctx,
                status="success",
            )

        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec2),
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            edit="quant_rtn",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )


@pytest.mark.parametrize("tier", ["fast", "turbo"])  # invalid tiers
def test_invalid_tier_exit(tmp_path: Path, tier):
    cfg = _write_cfg(tmp_path)
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )

        def runner_exec3(**kwargs):
            cfg_ctx = getattr(kwargs.get("config"), "context", {})
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context=cfg_ctx,
                status="success",
            )

        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec3),
            )
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                tier=tier,
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


@pytest.mark.parametrize("probes", [-1, 11])
def test_invalid_probes_exit(tmp_path: Path, probes):
    cfg = _write_cfg(tmp_path)
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )

        def runner_exec4(**kwargs):
            cfg_ctx = getattr(kwargs.get("config"), "context", {})
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context=cfg_ctx,
                status="success",
            )

        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec4),
            )
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                probes=probes,
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_probes_override_applied(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    captured = {}

    def runner_exec(**kwargs):
        captured["auto_config"] = kwargs.get("auto_config")
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            probes=3,
            tier="balanced",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert captured["auto_config"]["probes"] == 3


def test_invariants_injected_into_policy(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    captured = {}

    def detect_with_invariants(model_id, adapter):
        return SimpleNamespace(
            default_loss="ce",
            model_id=model_id,
            adapter=adapter,
            module_selectors={},
            invariants={"a", "b"},
            cert_lints=[],
            family="gpt",
        )

    def runner_exec(**kwargs):
        captured["run_config"] = kwargs.get("config")
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.detect_model_profile",
                detect_with_invariants,
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    policy = captured["run_config"].context["guards"]["invariants"]
    assert isinstance(policy.get("profile_checks", []), list)


def test_tokenizer_digest_unknown_path(tmp_path: Path):
    cfg = _write_cfg(tmp_path)

    class Tok:
        def vocab(self):  # not mapping
            return 5

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.resolve_tokenizer",
                lambda profile: (Tok(), None),
            )
        )

        def runner_exec5(**kwargs):
            cfg_ctx = getattr(kwargs.get("config"), "context", {})
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context=cfg_ctx,
                status="success",
            )

        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec5),
            )
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )


def test_mlm_mask_prob_zero_sets_labels_and_zero_counts(tmp_path: Path):
    cfg = _write_cfg(tmp_path, loss_type="mlm")
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.resolve_tokenizer",
                lambda profile: (
                    SimpleNamespace(
                        mask_token_id=103,
                        eos_token="</s>",
                        pad_token="</s>",
                        vocab_size=50000,
                    ),
                    "tokhash123",
                ),
            )
        )

        # Force mask_prob 0 via load_config replacement
        class Cfg:
            def __init__(self):
                self.model = SimpleNamespace(
                    adapter="hf_causal", id="gpt2", device="cpu"
                )
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
                self.eval = SimpleNamespace(
                    spike_threshold=2.0, loss=SimpleNamespace(type="mlm", mask_prob=0.0)
                )
                self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
                self.output = SimpleNamespace(dir=tmp_path / "runs")

            def model_dump(self):
                return {}

        stack.enter_context(
            patch("invarlock.core.config_loader.load_config", lambda p: Cfg())
        )

        def runner_exec6(**kwargs):
            cfg_ctx = getattr(kwargs.get("config"), "context", {})
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context=cfg_ctx,
                status="success",
            )

        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec6),
            )
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )


def test_mlm_missing_mask_token_exits(tmp_path: Path):
    cfg = _write_cfg(tmp_path, loss_type="mlm")
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        # No mask_token_id
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.resolve_tokenizer",
                lambda profile: (
                    SimpleNamespace(
                        eos_token="</s>", pad_token="</s>", vocab_size=50000
                    ),
                    "tokhash123",
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )

        def runner_exec7(**kwargs):
            cfg_ctx = getattr(kwargs.get("config"), "context", {})
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context=cfg_ctx,
                status="success",
            )

        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec7),
            )
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


@pytest.mark.parametrize(
    "field,base,conf",
    [
        ("seq_len", 4, 8),
        ("stride", 1, 4),
    ],
)
def test_baseline_meta_mismatch_exit(tmp_path: Path, field, base, conf):
    cfg = _write_cfg(tmp_path)
    meta = {
        "tokenizer_hash": "tokhash123",
        field: base,
        "dataset": "synthetic",
        "split": "validation",
    }
    baseline = _baseline_with_meta(tmp_path, meta, [[1, 2, 3]], [[4, 5, 6]])
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
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
                        evaluation_windows={},
                        status="success",
                    )
                ),
            )
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile="release",
                baseline=str(baseline),
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_baseline_dataset_split_mismatch_exit(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    meta = {"tokenizer_hash": "tokhash123", "dataset": "other", "split": "test"}
    baseline = _baseline_with_meta(tmp_path, meta, [[1, 2, 3]], [[4, 5, 6]])
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
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
                        evaluation_windows={},
                        status="success",
                    )
                ),
            )
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile="release",
                baseline=str(baseline),
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_release_baseline_missing_eval_windows_exit(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"meta": {"tokenizer_hash": "tokhash123"}}))
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile="release",
                baseline=str(baseline),
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_runner_context_none_is_coerced(tmp_path: Path):
    cfg = _write_cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context=None,
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
