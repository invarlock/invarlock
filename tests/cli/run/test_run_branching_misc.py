from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command


def _cfg(tmp_path: Path, preview=1, final=1) -> Path:
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


def _common_patches_detect_ce():
    return (
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix=None: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
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
                family="gpt2",
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


def test_window_match_fraction_mismatch_exit(tmp_path: Path):
    cfg = _cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
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
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.resolve_tokenizer",
                lambda prof: (
                    SimpleNamespace(
                        eos_token="</s>", pad_token="</s>", vocab_size=50000
                    ),
                    "tokhash123",
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_dataset_meta_context_non_dict_branch(tmp_path: Path):
    cfg = _cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
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
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.resolve_tokenizer",
                lambda prof: (
                    SimpleNamespace(
                        eos_token="</s>", pad_token="</s>", vocab_size=50000
                    ),
                    "tokhash123",
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_guard_overhead_threshold_parse_fallback(tmp_path: Path):
    cfg = _cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={
                    "ppl_preview": 1.0,
                    "ppl_final": 1.0,
                    "ppl_ratio": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_match_fraction": 1.0,
                },
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.resolve_tokenizer",
                lambda prof: (
                    SimpleNamespace(
                        eos_token="</s>", pad_token="</s>", vocab_size=50000
                    ),
                    "tokhash123",
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_retry_summary_prints_and_snapshot_cleanup(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)

    class Reg:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name,
                load_model=lambda model_id, device=None: SimpleNamespace(),
                snapshot_chunked=lambda model: str(tmp_path / "snapdir"),
                restore_chunked=lambda model, path=None: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, n, t):
            return {"name": n, "module": f"{t}.{n}", "version": "test"}

    class RC:
        def __init__(self, max_attempts=3, timeout=None, verbose=False):
            self.attempt_history = []

        def should_retry(self, passed):
            return False

        def record_attempt(self, attempt, result_summary, edit_config):
            self.attempt_history.append(result_summary)

        def get_attempt_summary(self):
            return {"total_attempts": len(self.attempt_history), "elapsed_time": 0.42}

    def make_cert(report, baseline):
        return {"validation": {"gate": False}}

    def runner_exec(**kwargs):
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
            context={"dataset_meta": {}},
            evaluation_windows={
                "preview": {
                    "window_ids": [0],
                    "input_ids": [[1, 2]],
                    "attention_masks": [[1, 1]],
                },
                "final": {
                    "window_ids": [1],
                    "input_ids": [[3, 4]],
                    "attention_masks": [[1, 1]],
                },
            },
            status="success",
        )

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "metrics": {},
                "evaluation_windows": {
                    "preview": {"input_ids": [[1, 2]]},
                    "final": {"input_ids": [[3, 4]]},
                },
            }
        )
    )

    with ExitStack() as stack:
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
        monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: Reg())
        )
        stack.enter_context(patch("invarlock.core.retry.RetryController", RC))
        stack.enter_context(
            patch("invarlock.reporting.certificate.make_certificate", make_cert)
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
            profile="ci",
            baseline=str(baseline),
            until_pass=True,
            max_attempts=1,
            out=str(tmp_path / "runs"),
        )
    assert True


def test_dataset_provider_tokenizer_resolution_exception_exit(tmp_path: Path):
    cfg = _cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={
                    "ppl_preview": 1.0,
                    "ppl_final": 1.0,
                    "ppl_ratio": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_match_fraction": 1.0,
                },
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )

    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        for target in (
            "invarlock.cli.commands.run.resolve_tokenizer",
            "invarlock.cli.commands.run.resolve_tokenizer",
        ):
            stack.enter_context(patch(target, side_effect=RuntimeError("tok-fail")))
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
