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
)
from tests.cli.run._support_run_common import (
    write_base_run_config as _base_cfg,
)


def _common_ce():
    return common_ce_patches(include_registry=True)


def test_baseline_tokenizer_hash_mismatch_exit(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash-OLD"},
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [1], "input_ids": [[4, 5, 6]]},
                },
            }
        )
    )

    with ExitStack() as stack:
        for ctx in _common_ce():
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
        # Force tokenizer hash to be 'tokhash123' so a mismatch with baseline occurs
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
                profile="release",
                out=str(tmp_path / "runs"),
                baseline=str(baseline),
                until_pass=False,
            )


def test_preview_final_tokens_computed_when_missing_in_baseline_meta(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [1], "input_ids": [[4]]},
                },
            }
        )
    )

    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[9, 9, 9]], attention_masks=[[1, 1, 1]]
                        ),
                        SimpleNamespace(input_ids=[[8]], attention_masks=[[1]]),
                    )
                ),
            )
        )

        def runner_factory():
            class R:
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
                            "window_pairing_reason": None,
                            "paired_windows": 1,
                            "loss_type": "ce",
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
                                "input_ids": [[4]],
                                "attention_masks": [[1]],
                            },
                        },
                        status="success",
                    )

            return R()

        stack.enter_context(patch("invarlock.core.runner.CoreRunner", runner_factory))
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

    data = captured["r"]["data"]
    assert data.get("preview_total_tokens") == 3
    assert data.get("final_total_tokens") == 1


def test_provider_indices_fallback_iteration(tmp_path: Path):
    # Provider returns indices iterable that is convertible to list; ensure run doesn’t crash
    cfg = _base_cfg(tmp_path, 1, 1)

    class Provider:
        def windows(self, **kwargs):
            prev = SimpleNamespace(
                input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]], indices=(0,)
            )
            fin = SimpleNamespace(
                input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]], indices=(1,)
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
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )
