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


def _base_cfg(tmp_path: Path, preview=1, final=1) -> Path:
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
  spike_threshold: 2.0
  loss:
    type: auto

output:
  dir: runs
        """
    )
    return p


def _common_ce():
    return (
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
        patch(
            "invarlock.cli.commands.run.resolve_tokenizer",
            lambda profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
    )


def _provider_simple():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )
    )


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
                self.model = SimpleNamespace(id="gpt2", adapter="hf_causal", device="cpu")
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
        stack.enter_context(patch("invarlock.cli.config.load_config", load_cfg))
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
                "invarlock.cli.commands.run.shutil.rmtree",
                side_effect=RuntimeError("boom"),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


@pytest.mark.parametrize(
    "percent,ratio,expect_na",
    [
        (None, 1.02, False),
        (math.nan, 1.01, False),
        (None, None, True),
    ],
)
def test_overhead_display_fallbacks(tmp_path: Path, percent, ratio, expect_na):
    cfg = _base_cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            if kwargs.get("guards") == []:
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
            checks={},
            metrics={"overhead_percent": percent, "overhead_ratio": ratio},
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        for target in (
            "invarlock.reporting.validate.validate_guard_overhead",
            "invarlock.cli.commands.run.validate_guard_overhead",
            "invarlock.cli.commands.run.validate_guard_overhead",
        ):
            stack.enter_context(patch(target, vg))
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        run_command(
            config=str(cfg), device="cpu", profile="ci", out=str(tmp_path / "runs")
        )


def test_baseline_schedule_skips_provider_windows(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
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
    baseline.write_text(json.dumps({"meta": {"tokenizer_hash": "tokhash123"}}))

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
            patch("invarlock.reporting.certificate.make_certificate", make_cert)
        )
        for target in (
            "invarlock.reporting.validate.validate_guard_overhead",
            "invarlock.cli.commands.run.validate_guard_overhead",
            "invarlock.cli.commands.run.validate_guard_overhead",
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
                        metrics={"overhead_ratio": 0.0, "overhead_percent": 0.0},
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

    assert attempts["cert"] == 1


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
                self.model = SimpleNamespace(id="gpt2", adapter="hf_causal", device="cpu")
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
        stack.enter_context(patch("invarlock.cli.config.load_config", load_cfg))
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


def test_guard_overhead_failure_exits(tmp_path: Path):
    # Validator returns passed=False → should exit when measure_guard_overhead is enabled (ci profile)
    cfg = _base_cfg(tmp_path)

    class Runner:
        def __init__(self):
            self.calls = 0

        def execute(self, **kwargs):
            self.calls += 1
            if kwargs.get("guards") == []:
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
        # Fail with overhead 5% vs threshold 1% and provide ratio to ensure evaluation
        return SimpleNamespace(
            passed=False,
            messages=[],
            warnings=[],
            errors=[],
            checks={},
            metrics={
                "overhead_percent": 5.0,
                "overhead_ratio": 1.05,
                "overhead_threshold": 0.01,
            },
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        for target in (
            "invarlock.reporting.validate.validate_guard_overhead",
            "invarlock.cli.commands.run.validate_guard_overhead",
            "invarlock.cli.commands.run.validate_guard_overhead",
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


def test_overhead_threshold_bad_type_uses_default(tmp_path: Path):
    # overhead_threshold not a float → fallback to GUARD_OVERHEAD_THRESHOLD internally, no crash
    cfg = _base_cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            if kwargs.get("guards") == []:
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
            checks={},
            metrics={"overhead_percent": 0.5, "overhead_threshold": "bad"},
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        for target in (
            "invarlock.reporting.validate.validate_guard_overhead",
            "invarlock.cli.commands.run.validate_guard_overhead",
        ):
            stack.enter_context(patch(target, vg))
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        run_command(
            config=str(cfg), device="cpu", profile="ci", out=str(tmp_path / "runs")
        )


# --- Selected serialization/OptionInfo/psutil/env fallback edge tests ---


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
                "invarlock.cli.commands.run.psutil.virtual_memory",
                lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider", lambda *a, **k: _provider_simple()
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_save_report_failure_bubbles_to_exit(tmp_path: Path):
    cfg = _base_cfg(tmp_path)

    def bad_save(*args, **kwargs):
        raise RuntimeError("cannot save")

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", bad_save))
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
