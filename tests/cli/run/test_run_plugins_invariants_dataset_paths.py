from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command
from tests.cli.run._support_run_common import configure_guard_metric_impact_skip
from tests.cli.run._support_run_common import (
    synthetic_provider_min as _provider_min,
)
from tests.cli.run._support_run_plugins import (
    plugins_invariants_baseline_with_meta as _baseline_with_meta,
)
from tests.cli.run._support_run_plugins import (
    plugins_invariants_common_ce as _common_ce,
)
from tests.cli.run._support_run_plugins import (
    plugins_invariants_write_cfg as _write_cfg,
)


def test_dedupe_reduction_raises_when_below_floor(tmp_path: Path):
    cfg = _write_cfg(tmp_path, preview=2, final=2)

    class Provider:
        def windows(self, **kwargs):
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


def test_ci_no_estimate_capacity_uses_default_window_plan(tmp_path: Path):
    cfg = configure_guard_metric_impact_skip(_write_cfg(tmp_path, preview=1, final=1))

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
        stack.enter_context(
            patch("invarlock.reporting.report_bundle.save_report", cap_save)
        )
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
            profile="ci",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )
    assert captured["r"]["data"]["window_plan"]["profile"] == "ci"


def test_dataset_hash_constructed_when_missing(tmp_path: Path):
    cfg = configure_guard_metric_impact_skip(_write_cfg(tmp_path, 1, 1))
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
        stack.enter_context(
            patch("invarlock.reporting.report_bundle.save_report", cap_save)
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
        stack.enter_context(
            patch("invarlock.reporting.report_bundle.save_report", cap_save)
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
        assert captured["r"]["metrics"].get("loss_type") in {"causal", "ce"}


def test_snapshot_auto_prefers_bytes_when_supported(tmp_path: Path):
    cfg = _write_cfg(tmp_path)

    class Adapter:
        name = "hf_causal"

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
                        evaluation_windows={
                            "final": {
                                "window_ids": [1],
                                "logloss": [0.0],
                                "token_counts": [1],
                            }
                        },
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
