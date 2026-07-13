from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command
from tests.cli.run._support_run_common import (
    assert_single_run_output_artifacts,
    common_ce_patches,
    write_base_run_config,
)
from tests.cli.run._support_run_common import (
    runner_success as _runner_success,
)
from tests.cli.run._support_run_common import (
    synthetic_provider_min as _provider_simple,
)


def _base_cfg(tmp_path: Path, preview=1, final=1) -> Path:
    return write_base_run_config(
        tmp_path,
        preview,
        final,
        eval_fields="  spike_threshold: 2.0\n",
    )


def _common_ce():
    return common_ce_patches(include_profile=False, include_save_report=True)


def _is_bare_control(kwargs: dict[str, object]) -> bool:
    cfg = kwargs.get("config")
    context = getattr(cfg, "context", None)
    if not isinstance(context, dict):
        return False
    validation = context.get("validation")
    return (
        isinstance(validation, dict)
        and validation.get("guard_metric_impact_mode") == "bare"
    )


# --- Selected serialization/OptionInfo/psutil/env fallback edge tests ---


def test_degradation_limit_bad_type_fails_closed(tmp_path: Path):
    cfg = _base_cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            if _is_bare_control(kwargs):
                return SimpleNamespace(
                    edit={},
                    metrics={"ppl_final": 1.0},
                    guards={},
                    context={"dataset_meta": {}},
                    status="success",
                )
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    def vg(*a, **k):
        return SimpleNamespace(
            passed=True,
            messages=[],
            warnings=[],
            errors=[],
            checks={"guard_metric_impact": True},
            metrics={
                "degradation": 1.005,
                "display_value": 0.5,
                "degradation_limit": "bad",
            },
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        for target in (
            "invarlock.reporting.validate.validate_guard_metric_impact",
            "invarlock.cli.run_runtime_exec.validate_guard_metric_impact",
        ):
            stack.enter_context(patch(target, vg))
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        with pytest.raises(click.exceptions.Exit) as exc_info:
            run_command(
                config=str(cfg),
                device="cpu",
                profile="ci",
                out=str(tmp_path / "runs"),
            )
        assert exc_info.value.exit_code == 1
    assert_single_run_output_artifacts(tmp_path)


def test_typer_optioninfo_import_failure(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        import sys

        stack.enter_context(patch.dict(sys.modules, {"typer.models": None}))
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
            profile=None,
            out=str(tmp_path / "runs"),
            edit=None,
            tier=None,
            probes=0,
            until_pass=False,
            max_attempts=1,
            timeout=None,
            baseline=None,
        )
    assert_single_run_output_artifacts(tmp_path)


def test_baseline_json_decode_error_fallback(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{ invalid json ")
    from json import JSONDecodeError

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        import json as _json

        orig_load = _json.load

        def conditional_bad_load(f):
            fname = getattr(f, "name", "")
            if isinstance(fname, str) and fname.endswith("baseline.json"):
                raise JSONDecodeError("bad", "{", 1)
            return orig_load(f)

        stack.enter_context(patch("json.load", conditional_bad_load))
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
        )
    assert_single_run_output_artifacts(tmp_path)


def test_psutil_virtual_memory_failure(tmp_path: Path):
    cfg = _base_cfg(tmp_path)

    class Adapter:
        name = "hf_causal"

        def __init__(self):
            self.restored = 0

        def load_model(self, model_id, device=None):
            return SimpleNamespace(
                named_parameters=lambda: [], named_buffers=lambda: []
            )

        def snapshot(self, model):
            return b"x"

        def restore(self, model, blob):
            self.restored += 1

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
                "invarlock.cli.run_runtime_exec.psutil.virtual_memory",
                lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", _runner_success))
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
    assert_single_run_output_artifacts(tmp_path)


def test_save_report_failure_bubbles_to_exit(tmp_path: Path):
    cfg = _base_cfg(tmp_path)

    def bad_save(*args, **kwargs):
        raise RuntimeError("cannot save")

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.reporting.report_bundle.save_report", bad_save)
        )
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
        with pytest.raises(click.exceptions.Exit):
            run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
