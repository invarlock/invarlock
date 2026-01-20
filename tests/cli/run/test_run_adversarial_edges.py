from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command


def _cfg(tmp_path: Path, preview=2, final=2) -> Path:
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
  order: []

eval:
  loss:
    type: auto

output:
  dir: runs
        """,
        encoding="utf-8",
    )
    return p


def _common_ce():
    return (
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.eval.data.get_provider",
            lambda *a, **k: SimpleNamespace(
                windows=lambda **kw: (
                    SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
                    SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
                )
            ),
        ),
        patch(
            "invarlock.cli.commands.run.resolve_tokenizer",
            lambda prof: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
    )


def test_overhead_percent_display_release_profile(tmp_path: Path):
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

    class Overhead:
        def __init__(self):
            self.passed = True
            self.messages = []
            self.warnings = []
            self.errors = []
            self.checks = {}
            self.metrics = {"overhead_percent": 2.5, "overhead_ratio": None}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.validate_guard_overhead",
                lambda *a, **k: Overhead(),
            )
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        run_command(
            config=str(cfg), device="cpu", profile="release", out=str(tmp_path / "runs")
        )


def test_counts_mismatch_exit_after_stratification(tmp_path: Path):
    cfg = _cfg(tmp_path, preview=2, final=2)
    from invarlock.eval.data import EvaluationWindow

    class Provider:
        def windows(self, **kwargs):
            # Return mis-matched counts to trigger parity exit
            prev = EvaluationWindow(
                input_ids=[[1, 2]], attention_masks=[[1, 1]], indices=[0]
            )
            fin = EvaluationWindow(input_ids=[[3]], attention_masks=[[1]], indices=[1])
            return prev, fin

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={},
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
        stack.enter_context(patch("invarlock.cli.device.resolve_device", lambda d: d))
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )
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
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        run_command(
            config=str(cfg), device="cpu", profile="ci", out=str(tmp_path / "runs")
        )


def test_save_report_missing_json_key_exits(tmp_path: Path):
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

    def bad_save(report, run_dir, formats=None, filename_prefix=None):
        return {"html": str(run_dir / "report.html")}

    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(patch("invarlock.reporting.report.save_report", bad_save))
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_release_planner_no_adjustment_branch():
    from invarlock.cli.commands.run import _plan_release_windows

    cap = {
        "available_unique": 2000,
        "available_nonoverlap": 2000,
        "total_tokens": 1_000_000,
        "dedupe_rate": 0.05,
    }
    plan = _plan_release_windows(
        cap,
        requested_preview=300,
        requested_final=300,
        max_calibration=24,
        console=None,
    )
    assert plan["actual_preview"] == 300 and plan["coverage_ok"] is True


def test_release_planner_candidate_limit_only_branch():
    from invarlock.cli.commands.run import _plan_release_windows

    cap = {
        "available_unique": 2000,
        "available_nonoverlap": 2000,
        "total_tokens": 1_000_000,
        "dedupe_rate": 0.05,
        "candidate_limit": 1000,
    }
    plan = _plan_release_windows(
        cap,
        requested_preview=400,
        requested_final=400,
        max_calibration=24,
        console=None,
    )
    assert plan["capacity"].get("candidate_limit") == 1000


def test_snapshot_auto_bytes_and_chunked_paths(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)

    class LargeModel:
        def named_parameters(self):
            return [
                (
                    "p",
                    SimpleNamespace(
                        element_size=lambda: 1, nelement=lambda: 900_000_000
                    ),
                )
            ]

        def named_buffers(self):
            return []

    class Adapter:
        name = "hf_causal"

        def __init__(self):
            self.snapshots = []

        def load_model(self, model_id, device=None):
            return LargeModel()

        def snapshot(self, model):
            self.snapshots.append("bytes")
            return b"x"

        def restore(self, model, blob):
            return None

        def snapshot_chunked(self, model):
            self.snapshots.append("chunked")
            return str(tmp_path / "snapdir")

        def restore_chunked(self, model, path):
            return None

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.psutil.virtual_memory",
                lambda: SimpleNamespace(available=50 * 1024 * 1024),
            )
        )  # 50MB
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.shutil.disk_usage",
                lambda path: SimpleNamespace(
                    total=0, used=0, free=10 * 1024 * 1024 * 1024
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.registry.get_registry",
                lambda: SimpleNamespace(
                    get_adapter=lambda n: Adapter(),
                    get_edit=lambda n: SimpleNamespace(name=n),
                    get_guard=lambda n: SimpleNamespace(name=n),
                    get_plugin_metadata=lambda n, t: {
                        "name": n,
                        "module": f"{t}.{n}",
                        "version": "test",
                    },
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
                        evaluation_windows={},
                        status="success",
                    )
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_bytes_only_adapter_path(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id, device=None):
            class M:
                def named_parameters(self):
                    return []

                def named_buffers(self):
                    return []

            return M()

        def snapshot(self, model):
            return b"x"

        def restore(self, model, blob):
            return None

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.core.registry.get_registry",
                lambda: SimpleNamespace(
                    get_adapter=lambda n: Adapter(),
                    get_edit=lambda n: SimpleNamespace(name=n),
                    get_guard=lambda n: SimpleNamespace(name=n),
                    get_plugin_metadata=lambda n, t: {
                        "name": n,
                        "module": f"{t}.{n}",
                        "version": "test",
                    },
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
                        evaluation_windows={},
                        status="success",
                    )
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_chunked_only_adapter_path(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id, device=None):
            class M:
                def named_parameters(self):
                    return []

                def named_buffers(self):
                    return []

            return M()

        def snapshot_chunked(self, model):
            return str(tmp_path / "snapdir")

        def restore_chunked(self, model, path):
            return None

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.core.registry.get_registry",
                lambda: SimpleNamespace(
                    get_adapter=lambda n: Adapter(),
                    get_edit=lambda n: SimpleNamespace(name=n),
                    get_guard=lambda n: SimpleNamespace(name=n),
                    get_plugin_metadata=lambda n, t: {
                        "name": n,
                        "module": f"{t}.{n}",
                        "version": "test",
                    },
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
                        evaluation_windows={},
                        status="success",
                    )
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_top_level_exception_with_debug_trace(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            raise RuntimeError("boom")

    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
        for ctx in _common_ce():
            stack.enter_context(ctx)
        monkeypatch.setenv("INVARLOCK_DEBUG_TRACE", "1")
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_tokenizer_digest_nonstring_keys_fallback(tmp_path: Path):
    cfg = _cfg(tmp_path)

    class Tok:
        def get_vocab(self):
            return {1: "a", 2: "b"}

    def resolver(_):
        return Tok(), None

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.commands.run.resolve_tokenizer", resolver)
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
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
