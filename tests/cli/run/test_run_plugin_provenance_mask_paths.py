from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from invarlock.cli.commands.run import run_command
from tests.cli.run._support_run_common import (
    assert_single_run_output_artifacts,
    common_ce_patches,
)
from tests.cli.run._support_run_common import (
    write_base_run_config as _write_base_run_config,
)


def _cfg(tmp_path: Path, preview=4, final=4) -> Path:
    return _write_base_run_config(
        tmp_path,
        preview,
        final,
        edit_name="structured",
        eval_fields="  spike_threshold: 2.0\n",
    )


def _common_ce():
    return common_ce_patches(
        include_registry=True,
        include_save_report=True,
        tokenizer_vocab_size=1000,
    )


def _provider_simple():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]]),
            SimpleNamespace(input_ids=[[5, 6, 7, 8]], attention_masks=[[1, 1, 1, 1]]),
        )
    )


def _is_bare_control(kwargs: dict[str, object]) -> bool:
    cfg = kwargs.get("config")
    context = getattr(cfg, "context", None)
    if not isinstance(context, dict):
        return False
    validation = context.get("validation")
    return (
        isinstance(validation, dict) and validation.get("guard_overhead_mode") == "bare"
    )


def test_baseline_attention_masks_inferred_and_labels_sanitized(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 0, 2]],
                        "labels": [[1]],
                    },
                    "final": {"window_ids": [1], "input_ids": [[3, 4, 5]]},
                },
            }
        )
    )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        # Runner must provide evaluation_windows to satisfy release+baseline, but we will run in default profile
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
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
        run_command(
            config=str(cfg),
            device="cpu",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
            until_pass=False,
        )
    assert_single_run_output_artifacts(tmp_path)


def test_baseline_labels_longer_than_input_trimmed(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    # Labels longer than input_ids should be trimmed safely
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {
                        "window_ids": [0],
                        "input_ids": [[1, 2]],
                        "labels": [[1, 2, 3, 4, 5]],
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[3, 4]],
                        "labels": [[6, 7, 8]],
                    },
                },
            }
        )
    )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
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
        # No exception implies trimming path executed safely
        run_command(
            config=str(cfg),
            device="cpu",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
            until_pass=False,
        )
    assert_single_run_output_artifacts(tmp_path)


def test_provider_attention_mask_tolist_tuple_path(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)

    class AM:
        def __init__(self, n):
            self.n = n

        def tolist(self):
            return tuple([1] * self.n)

    class Provider:
        def windows(self, **kwargs):
            return (
                SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[AM(3)]),
                SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[AM(3)]),
            )

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
    assert_single_run_output_artifacts(tmp_path)


@pytest.mark.parametrize(
    "key",
    [
        "window_pairing_reason",
        "window_pairing_preview",
        "window_pairing_final",
        "paired_windows",
        "paired_delta_summary",
    ],
)
def test_metrics_optional_pairing_fields_passthrough(tmp_path: Path, key: str):
    cfg = _cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    runner_metrics = {
        "ppl_preview": 1.0,
        "ppl_final": 1.0,
        "ppl_ratio": 1.0,
        key: {"ok": True} if key.endswith("summary") else 1,
    }
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.reporting.report_files.save_report", cap_save)
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics=runner_metrics,
                        guards={},
                        context={"dataset_meta": {}},
                        status="success",
                    )
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
    assert key in captured["r"].get("metrics", {})


@pytest.mark.parametrize(
    "opt_key",
    [
        "algorithm_version",
        "implementation",
        "scope",
        "ranking",
        "grouping",
        "budgets",
        "seed",
        "mask_digest",
    ],
)
def test_edit_optional_fields_transfer(tmp_path: Path, opt_key: str):
    cfg = _cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    edit_payload = {opt_key: "X"}
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.reporting.report_files.save_report", cap_save)
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit=edit_payload,
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
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
    assert opt_key in captured["r"]["edit"]
