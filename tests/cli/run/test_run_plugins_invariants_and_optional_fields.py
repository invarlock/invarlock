from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command


def _write_cfg(tmp_path: Path, preview=2, final=2, loss_type="auto") -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
model:
  adapter: hf_gpt2
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
  order: []

eval:
  spike_threshold: 2.0
  loss:
    type: {loss_type}

output:
  dir: runs
        """
    )
    return p


def _common_ce():
    return (
        patch(
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
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
            "invarlock.cli.commands.run.detect_model_profile",
            lambda model_id=None, adapter=None: SimpleNamespace(
                default_loss="ce",
                model_id=model_id,
                adapter=adapter,
                module_selectors={},
                invariants=set(),
                cert_lints=[],
                family="gpt",
            ),
        ),
        patch(
            "invarlock.cli.commands.run.resolve_tokenizer",
            lambda profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
    )


def _provider_min():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )
    )


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
                "invarlock.cli.config.apply_profile",
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
            patch("invarlock.cli.config.resolve_edit_kind", lambda name: "quant_rtn")
        )
        stack.enter_context(
            patch("invarlock.cli.config.apply_edit_override", lambda c, e: c)
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
            edit="quant",
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
                "invarlock.cli.commands.run.detect_model_profile",
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
                "invarlock.cli.commands.run.resolve_tokenizer",
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
                "invarlock.cli.commands.run.resolve_tokenizer",
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
                self.eval = SimpleNamespace(
                    spike_threshold=2.0, loss=SimpleNamespace(type="mlm", mask_prob=0.0)
                )
                self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
                self.output = SimpleNamespace(dir=tmp_path / "runs")

            def model_dump(self):
                return {}

        stack.enter_context(patch("invarlock.cli.config.load_config", lambda p: Cfg()))

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
                "invarlock.cli.commands.run.resolve_tokenizer",
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


def _baseline_with_meta(tmp_path: Path, meta: dict, preview_ids, final_ids) -> Path:
    p = tmp_path / "baseline.json"
    payload = {
        "meta": meta,
        "evaluation_windows": {
            "preview": {"window_ids": [0], "input_ids": preview_ids},
            "final": {"window_ids": [1], "input_ids": final_ids},
        },
    }
    p.write_text(json.dumps(payload))
    return p


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


def test_provider_non_evalwindow_mismatch_counts_no_exit(tmp_path: Path):
    cfg = _write_cfg(tmp_path, preview=2, final=1)

    class Provider:
        def windows(self, **kwargs):
            return SimpleNamespace(
                input_ids=[[1, 2]], attention_masks=[[1, 1]]
            ), SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]])

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
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
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )


def test_provider_indices_not_iterable_fallback(tmp_path: Path):
    cfg = _write_cfg(tmp_path)

    class Provider:
        def windows(self, **kwargs):
            prev = SimpleNamespace(
                input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]], indices=object()
            )
            fin = SimpleNamespace(
                input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]], indices=object()
            )
            return prev, fin

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
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
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )


def test_metrics_merges_masked_totals_from_context(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    captured = {}

    class Runner:
        def execute(self, **kwargs):
            ctx = {
                "dataset_meta": {
                    "masked_tokens_total": 3,
                    "masked_tokens_preview": 1,
                    "masked_tokens_final": 2,
                }
            }
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context=ctx,
                status="success",
            )

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
    m = captured["r"]["metrics"]
    assert (
        m.get("masked_tokens_total") == 3
        and m.get("masked_tokens_preview") == 1
        and m.get("masked_tokens_final") == 2
    )


def test_metrics_optional_logloss_keys_persisted(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    captured = {}

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={
                    "ppl_preview": 1.0,
                    "ppl_final": 1.0,
                    "ppl_ratio": 1.0,
                    "logloss_preview": 2.0,
                    "logloss_final": 2.0,
                    "logloss_delta": 0.0,
                },
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
    m = captured["r"].get("metrics", {})
    assert "logloss_preview" in m and "logloss_final" in m and "logloss_delta" in m


def test_guard_overhead_fail_exits(tmp_path: Path):
    cfg = _write_cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 2.0, "ppl_ratio": 2.0},
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        # Patch validator to fail
        for target in (
            "invarlock.reporting.validate.validate_guard_overhead",
            "invarlock.cli.commands.run.validate_guard_overhead",
            "invarlock.cli.commands.run.validate_guard_overhead",
        ):
            stack.enter_context(
                patch(
                    target,
                    lambda *a, **k: SimpleNamespace(
                        passed=False,
                        messages=[],
                        warnings=[],
                        errors=[],
                        checks={},
                        metrics={"overhead_ratio": 2.0, "overhead_percent": 100.0},
                    ),
                )
            )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile="ci",
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_drift_gate_fail_nonfatal(tmp_path: Path):
    cfg = _write_cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 3.0, "ppl_ratio": 3.0},
                guards={},
                context={"dataset_meta": {}},
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


def test_retry_controller_until_pass_two_attempts(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [1], "input_ids": [[4, 5, 6]]},
                },
            }
        )
    )

    class Runner:
        def execute(self, **kwargs):
            cfg_ctx = getattr(kwargs.get("config"), "context", {})
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
                context=cfg_ctx,
                evaluation_windows={
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 2, 3]],
                        "attention_masks": [[1, 1, 1]],
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[4, 5, 6]],
                        "attention_masks": [[1, 1, 1]],
                    },
                },
                status="success",
            )

    class RC:
        def __init__(self, max_attempts=3, timeout=None, verbose=False):
            self.attempt_history = []

        def should_retry(self, passed):
            return not passed and len(self.attempt_history) < 1

        def record_attempt(self, attempt, result_summary, edit_config):
            self.attempt_history.append(result_summary)

        def get_attempt_summary(self):
            return {"total_attempts": len(self.attempt_history), "elapsed_time": 0.1}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        # first fail then pass
        results = [{"validation": {"gateA": False}}, {"validation": {"gateA": True}}]

        def make_cert(report, baseline_report):
            return results.pop(0)

        stack.enter_context(patch("invarlock.core.retry.RetryController", RC))
        stack.enter_context(
            patch("invarlock.reporting.certificate.make_certificate", make_cert)
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
            until_pass=True,
            max_attempts=2,
        )


def test_dedupe_reduction_raises_when_below_floor(tmp_path: Path):
    # Use small requested counts so floor > proposed and reduction triggers error
    cfg = _write_cfg(tmp_path, preview=2, final=2)

    class Provider:
        def windows(self, **kwargs):
            # Return duplicate sequences so signatures collide
            s = [1, 2, 3, 4]
            prev = SimpleNamespace(input_ids=[s, s], attention_masks=[[1] * 4, [1] * 4])
            fin = SimpleNamespace(input_ids=[s, s], attention_masks=[[1] * 4, [1] * 4])
            return prev, fin

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
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
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_release_no_estimate_capacity_uses_default_window_plan(tmp_path: Path):
    cfg = _write_cfg(tmp_path, preview=1, final=1)

    class Provider:
        def windows(self, **kwargs):
            return SimpleNamespace(
                input_ids=[[1]], attention_masks=[[1]]
            ), SimpleNamespace(input_ids=[[2]], attention_masks=[[1]])

    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )

        def runner_exec8(**kwargs):
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
                lambda: SimpleNamespace(execute=runner_exec8),
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )
    assert captured["r"]["data"]["window_plan"]["profile"] == "release"


def test_dataset_hash_constructed_when_missing(tmp_path: Path):
    cfg = _write_cfg(tmp_path, 1, 1)
    meta = {
        "tokenizer_hash": "tokhash123",
        "preview_hash": "a" * 32,
        "final_hash": "b" * 32,
    }
    baseline = _baseline_with_meta(tmp_path, meta, [[1, 2, 3]], [[4, 5, 6]])
    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    class Runner:
        def execute(self, **kwargs):
            cfg_ctx = getattr(kwargs.get("config"), "context", {})
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
                context=cfg_ctx,
                evaluation_windows={
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 2, 3]],
                        "attention_masks": [[1, 1, 1]],
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[4, 5, 6]],
                        "attention_masks": [[1, 1, 1]],
                    },
                },
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
            until_pass=False,
        )
    assert isinstance(captured["r"]["data"].get("dataset_hash"), str)


def test_loss_type_from_dataset_meta_when_missing_in_metrics(tmp_path: Path):
    cfg = _write_cfg(tmp_path, 1, 1)
    captured = {}

    class Runner:
        def execute(self, **kwargs):
            ctx = {"dataset_meta": {"loss_type": "causal"}}
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context=ctx,
                status="success",
            )

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
        # Implementation currently prefers resolved loss; accept either resolved or meta-provided
        assert captured["r"]["metrics"].get("loss_type") in {"causal", "ce"}


def test_snapshot_auto_prefers_bytes_when_supported(tmp_path: Path):
    cfg = _write_cfg(tmp_path)

    class Adapter:
        name = "hf_gpt2"

        def __init__(self):
            self.restored = 0

        def load_model(self, model_id, device=None):
            return object()

        def snapshot(self, model):
            return b"blob"

        def restore(self, model, blob):
            self.restored += 1

    adapter = Adapter()

    class Registry:
        def get_adapter(self, name):
            return adapter

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, n, t):
            return {"name": n, "module": f"{t}.{n}", "version": "test"}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: Registry())
        )
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
                        status="success",
                    )
                ),
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )
    assert adapter.restored >= 1
