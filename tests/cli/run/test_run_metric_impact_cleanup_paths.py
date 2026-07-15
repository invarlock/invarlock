from __future__ import annotations

import json
import math
import os
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


def test_cleanup_rmtree_exception_is_swallowed(tmp_path: Path, monkeypatch):
    cfg = _base_cfg(tmp_path)

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id, device=None):
            return SimpleNamespace(
                named_parameters=lambda: [], named_buffers=lambda: []
            )

        def snapshot_chunked(self, model):
            return str(tmp_path / "snapdir")

        def restore_chunked(self, model, path):
            return None

    adapter = Adapter()

    def load_cfg(p):
        class Cfg:
            def __init__(self):
                self.model = SimpleNamespace(
                    id="gpt2", adapter="hf_causal", device="cpu"
                )
                self.edit = SimpleNamespace(name="quant_rtn", plan={})
                self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
                self.guards = SimpleNamespace(order=[])
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
                self.eval = SimpleNamespace(
                    spike_threshold=2.0, loss=SimpleNamespace(type="auto")
                )
                self.output = SimpleNamespace(dir=tmp_path / "runs")
                self.context = {"snapshot": {"mode": "chunked"}}

            def model_dump(self):
                return {}

        return Cfg()

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.core.config_loader.load_config", load_cfg))
        stack.enter_context(
            patch(
                "invarlock.core.registry.get_registry",
                lambda: SimpleNamespace(
                    get_adapter=lambda n: adapter,
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
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.checkpoint.shutil.rmtree",
                side_effect=RuntimeError("boom"),
            )
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", _runner_success))
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
    assert_single_run_output_artifacts(tmp_path)


@pytest.mark.parametrize(
    "display_value,degradation",
    [
        (None, 0.005),
        (math.nan, 0.01),
        (None, None),
    ],
)
def test_metric_impact_incomplete_measurements_fail_closed(
    tmp_path: Path, display_value, degradation
):
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
                "display_value": display_value,
                "degradation": degradation,
            },
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        for target in (
            "invarlock.reporting.validate.validate_guard_metric_impact",
            "invarlock.cli.run_runtime_exec.validate_guard_metric_impact",
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


def test_baseline_schedule_skips_provider_windows(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
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
            }
        )
    )

    called = {"windows": 0}

    class Provider:
        def windows(self, **kwargs):
            called["windows"] += 1
            raise AssertionError(
                "Provider.windows should not be called when using baseline schedule"
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
            config=str(cfg),
            device="cpu",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
        )

    assert called["windows"] == 0


def test_until_pass_baseline_disappears_between_attempts(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
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
            }
        )
    )

    attempts = {"exec": 0, "cert": 0}

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id, device=None):
            return object()

    def runner_exec(**kwargs):
        attempts["exec"] += 1
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )

    class RC:
        def __init__(self, max_attempts=3, timeout=None, verbose=False):
            self.attempt_history = []

        def should_retry(self, passed):
            return len(self.attempt_history) == 1

        def record_attempt(self, attempt, result_summary, edit_config):
            self.attempt_history.append(result_summary)

        def get_attempt_summary(self):
            return {"total_attempts": len(self.attempt_history), "elapsed_time": 0.1}

    def make_cert(report, baseline_report):
        attempts["cert"] += 1
        if attempts["cert"] == 1:
            try:
                os.remove(baseline)
            except OSError:
                pass
        return {"validation": {"primary_metric_acceptable": False}}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.core.retry.RetryController", RC))
        stack.enter_context(
            patch("invarlock.cli.run_execution.build_evaluation_report", make_cert)
        )
        for target in (
            "invarlock.reporting.validate.validate_guard_metric_impact",
            "invarlock.cli.run_runtime_exec.validate_guard_metric_impact",
            "invarlock.cli.run_runtime_exec.validate_guard_metric_impact",
        ):
            stack.enter_context(
                patch(
                    target,
                    lambda *a, **k: SimpleNamespace(
                        passed=True,
                        messages=[],
                        warnings=[],
                        errors=[],
                        checks={},
                        metrics={"degradation": 0.0, "display_value": 0.0},
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
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
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
            profile="dev",
            baseline=str(baseline),
            until_pass=True,
            max_attempts=2,
            out=str(tmp_path / "runs"),
        )

    assert attempts["cert"] == 2


def test_restore_chunked_missing_dir_causes_exit(tmp_path: Path):
    cfg = _base_cfg(tmp_path)

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id, device=None):
            return object()

        def snapshot_chunked(self, model):
            return str(tmp_path / "snap")

        def restore_chunked(self, model, path):
            raise FileNotFoundError("missing snapshot dir")

    adapter = Adapter()

    def load_cfg(p):
        class Cfg:
            def __init__(self):
                self.model = SimpleNamespace(
                    id="gpt2", adapter="hf_causal", device="cpu"
                )
                self.edit = SimpleNamespace(name="quant_rtn", plan={})
                self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
                self.guards = SimpleNamespace(order=[])
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
                self.eval = SimpleNamespace(
                    spike_threshold=2.0, loss=SimpleNamespace(type="auto")
                )
                self.output = SimpleNamespace(dir=tmp_path / "runs")
                self.context = {"snapshot": {"mode": "chunked"}}

            def model_dump(self):
                return {}

        return Cfg()

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.core.config_loader.load_config", load_cfg))
        stack.enter_context(
            patch(
                "invarlock.core.registry.get_registry",
                lambda: SimpleNamespace(
                    get_adapter=lambda n: adapter,
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
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_drift_boundary_precision_failure(tmp_path: Path):
    cfg = _base_cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={
                    "ppl_preview": 1.0,
                    "ppl_final": 1.0,
                    "ppl_ratio": 1.0500000000000001,
                },
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
    assert_single_run_output_artifacts(tmp_path)


def test_guard_metric_impact_failure_exits(tmp_path: Path):
    # Validator returns passed=False → should exit when measure_guard_metric_impact is enabled (ci profile)
    cfg = _base_cfg(tmp_path)

    class Runner:
        def __init__(self):
            self.calls = 0

        def execute(self, **kwargs):
            self.calls += 1
            if _is_bare_control(kwargs):
                # Bare run with ppl_final present
                return SimpleNamespace(
                    edit={},
                    metrics={"ppl_final": 1.0},
                    guards={},
                    context={"dataset_meta": {}},
                    status="success",
                )
            # Guarded run
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.02, "ppl_ratio": 1.02},
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    def vg(*a, **k):
        # Fail with 5% degradation against a 1% limit.
        return SimpleNamespace(
            passed=False,
            messages=[],
            warnings=[],
            errors=[],
            checks={},
            metrics={
                "display_value": 5.0,
                "degradation": 1.05,
                "degradation_limit": 0.01,
            },
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        for target in (
            "invarlock.reporting.validate.validate_guard_metric_impact",
            "invarlock.cli.run_runtime_exec.validate_guard_metric_impact",
            "invarlock.cli.run_runtime_exec.validate_guard_metric_impact",
        ):
            stack.enter_context(patch(target, vg))
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg), device="cpu", profile="ci", out=str(tmp_path / "runs")
            )
