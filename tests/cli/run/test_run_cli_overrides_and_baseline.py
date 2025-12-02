from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command


def _base_cfg(tmp_path: Path, preview=2, final=2) -> Path:
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
  loss:
    type: auto

output:
  dir: runs
        """
    )
    return p


def _common_ce():
    return (
        patch(
            "invarlock.cli.commands.run.detect_model_profile",
            lambda model_id, adapter: SimpleNamespace(
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
            lambda model_profile: (
                SimpleNamespace(
                    name_or_path="tok",
                    eos_token="</s>",
                    pad_token="</s>",
                    vocab_size=50000,
                ),
                "tokhash123",
            ),
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.core.registry.get_registry",
            lambda: SimpleNamespace(
                get_adapter=lambda name: SimpleNamespace(
                    name=name, load_model=lambda model_id, device: object()
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
    )


def _runner_min():
    return SimpleNamespace(
        execute=lambda **k: SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )
    )


def _runner_echo_context():
    def _exec(**kwargs):
        cfg_ctx = getattr(kwargs.get("config"), "context", {})
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context=cfg_ctx,
            evaluation_windows={},
            status="success",
        )

    return SimpleNamespace(execute=_exec)


def _runner_echo_context_with_eval(pre_ids, fin_ids):
    def _exec(**kwargs):
        cfg_ctx = getattr(kwargs.get("config"), "context", {})
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context=cfg_ctx,
            evaluation_windows={
                "preview": {
                    "window_ids": [0],
                    "input_ids": pre_ids,
                    "attention_masks": [[1] * len(pre_ids[0])],
                },
                "final": {
                    "window_ids": [1],
                    "input_ids": fin_ids,
                    "attention_masks": [[1] * len(fin_ids[0])],
                },
            },
            status="success",
        )

    return SimpleNamespace(execute=_exec)


def _provider_min():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )
    )


def test_auto_tier_override_conservative(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", _runner_echo_context)
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            tier="conservative",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert captured["r"]["meta"]["auto"]["tier"] == "conservative"


def test_auto_tier_override_aggressive(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", _runner_echo_context)
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            tier="aggressive",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert captured["r"]["meta"]["auto"]["tier"] == "aggressive"


def test_auto_probes_override_zero(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", _runner_echo_context)
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            probes=0,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert captured["r"]["meta"]["auto"]["probes"] == 0


def test_auto_probes_override_three(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", _runner_echo_context)
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            probes=3,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert captured["r"]["meta"]["auto"]["probes"] == 3


def test_apply_profile_raises_causes_exit(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.config.apply_profile",
                side_effect=RuntimeError("bad profile"),
            )
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", _runner_min))
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile="release",
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_data_meta_has_tokenizer_name(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", _runner_echo_context)
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert "tokenizer_name" in captured["r"]["data"]


def test_data_meta_has_tokenizer_hash(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", _runner_echo_context)
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert isinstance(captured["r"]["data"].get("tokenizer_hash"), str)


def test_data_meta_has_vocab_size_and_tokens(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", _runner_echo_context)
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    d = captured["r"]["data"]
    assert isinstance(d.get("vocab_size"), int) and d.get("vocab_size") > 0
    # eos/pad present (values may vary per tokenizer)
    assert d.get("eos_token") is not None and d.get("pad_token") is not None


def test_data_meta_add_prefix_space_key_present(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", _runner_echo_context)
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert "add_prefix_space" in captured["r"]["data"]


def test_metrics_overlap_triggers_exit(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={
                    "ppl_preview": 1.0,
                    "ppl_final": 1.0,
                    "ppl_ratio": 1.0,
                    "window_overlap_fraction": 0.1,
                    "window_match_fraction": 1.0,
                },
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )

    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )


def test_fallback_to_dataset_when_baseline_no_eval_windows_in_ci(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"meta": {"tokenizer_hash": "tokhash123"}}))

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", _runner_echo_context)
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
            until_pass=False,
        )


def test_preview_and_final_hash_and_dataset_hash_present(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    runner_eval = _runner_echo_context_with_eval([[1, 2, 3]], [[4, 5, 6]])

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", lambda: runner_eval)
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    d = captured["r"]["data"]
    assert isinstance(d.get("preview_hash"), str)
    assert isinstance(d.get("final_hash"), str)
    assert isinstance(d.get("dataset_hash"), str)


def test_window_plan_present_and_capacity_mirrored(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    class Provider:
        def estimate_capacity(self, **kwargs):
            return {
                "available_unique": 1000,
                "available_nonoverlap": 1000,
                "total_tokens": 1_000_000,
                "dedupe_rate": 0.1,
                "candidate_limit": 500,
            }

        def windows(self, **kwargs):
            return (
                SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
                SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", _runner_echo_context)
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    d = captured["r"]["data"]
    assert d.get("window_plan") and d["window_plan"]["profile"] == "release"
    assert (
        d.get("window_capacity") and d["window_capacity"].get("candidate_limit") == 500
    )


def test_baseline_masked_counts_propagated(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123", "window_plan": {}},
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 2, 3]],
                        "attention_masks": [[1, 1, 1]],
                        "masked_token_counts": [1],
                        "actual_token_counts": [3],
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[4, 5, 6]],
                        "attention_masks": [[1, 1, 1]],
                        "masked_token_counts": [2],
                        "actual_token_counts": [3],
                    },
                },
            }
        )
    )

    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context={"dataset_meta": {}},
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
        # Force MLM to enable masked token aggregation
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.detect_model_profile",
                lambda model_id, adapter: SimpleNamespace(
                    default_loss="mlm",
                    model_id=model_id,
                    adapter=adapter,
                    module_selectors={},
                    invariants=set(),
                    cert_lints=[],
                    family="bert",
                ),
            )
        )
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
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        runner_eval = _runner_echo_context_with_eval([[1, 2, 3]], [[4, 5, 6]])
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", lambda: runner_eval)
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    # Masked token totals live under report["data"] when runner/context preserves dataset_meta.
    # Accept optional presence; when present, it must be a positive integer.
    d = captured["r"].get("data", {})
    tot = d.get("masked_tokens_total")
    if tot is not None:
        assert isinstance(tot, int) and tot > 0


def test_labels_numpy_arrays_in_baseline(tmp_path: Path):
    import numpy as np

    cfg = _base_cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123", "window_plan": {}},
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 2, 3]],
                        "attention_masks": [[1, 1, 1]],
                        "labels": np.array([1, -100, -100]).tolist(),
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[4, 5, 6]],
                        "attention_masks": [[1, 1, 1]],
                        "labels": np.array([-100, -100, -100]).tolist(),
                    },
                },
            }
        )
    )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        runner_eval2 = _runner_echo_context_with_eval([[1, 2, 3]], [[4, 5, 6]])
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", lambda: runner_eval2)
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
            until_pass=False,
        )
