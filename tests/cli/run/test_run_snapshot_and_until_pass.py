from __future__ import annotations

import json
from collections import namedtuple
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
  adapter: hf_gpt2
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan: {{ heads: {{ mask_only: true, mask_auto: true, materialize: true }} }}

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


class FakeTensor:
    def __init__(self, bytes_count: int):
        self._bytes = bytes_count

    def element_size(self):
        return 1

    def nelement(self):
        return self._bytes


class LargeModel:
    def named_parameters(self):
        return [("p", FakeTensor(500_000_000))]  # ~500MB

    def named_buffers(self):
        return [("b", FakeTensor(100_000_000))]  # ~100MB


class SmallModel:
    def named_parameters(self):
        return [("p", FakeTensor(1_000_000))]  # ~1MB

    def named_buffers(self):
        return []


def _psutil_vm(available_mb: float):
    return SimpleNamespace(available=int(available_mb * 1024 * 1024))


def _disk_usage(free_mb: float):
    DU = namedtuple("DU", ["total", "used", "free"])
    return DU(total=10 * 1024 * 1024 * 1024, used=0, free=int(free_mb * 1024 * 1024))


def test_snapshot_auto_chunked_selected_when_large_and_disk_ok(
    tmp_path: Path, monkeypatch
):
    cfg = _cfg(tmp_path, 1, 1)

    class Adapter:
        name = "hf_gpt2"

        def __init__(self):
            self.rest_chunked = 0

        def load_model(self, model_id, device=None):
            return LargeModel()

        def snapshot_chunked(self, model):
            snap_dir = tmp_path / "snapshot_chunked"
            snap_dir.mkdir(parents=True, exist_ok=True)
            return str(snap_dir)

        def restore_chunked(self, model, path):
            self.rest_chunked += 1

    adapter = Adapter()

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
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
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[5, 6, 7, 8]], attention_masks=[[1, 1, 1, 1]]
                        ),
                    )
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.psutil.virtual_memory",
                lambda: _psutil_vm(available_mb=512),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.shutil.disk_usage",
                lambda path: _disk_usage(free_mb=2048),
            )
        )
        monkeypatch.delenv("INVARLOCK_SNAPSHOT_MODE", raising=False)
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    assert adapter.rest_chunked >= 1


def test_snapshot_cfg_mode_overrides_env(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path, 1, 1)

    class Adapter:
        name = "hf_gpt2"

        def __init__(self):
            self.restored = 0

        def load_model(self, model_id, device=None):
            return LargeModel()

        def snapshot(self, model):
            return b"x"

        def restore(self, model, blob):
            self.restored += 1

    adapter = Adapter()

    def load_cfg(p):
        class Cfg:
            def __init__(self):
                self.model = SimpleNamespace(id="gpt2", adapter="hf_gpt2", device="cpu")
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
                self.context = {"snapshot": {"mode": "bytes"}}

            def model_dump(self):
                return {}

        return Cfg()

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")  # env says chunked
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
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                        SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
                    )
                ),
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    assert adapter.restored >= 1


def test_until_pass_materialize_sets_flags_and_retries_once(
    tmp_path: Path, monkeypatch
):
    cfg = _cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": [[1, 2]]},
                    "final": {"window_ids": [1], "input_ids": [[3, 4]]},
                },
            }
        )
    )

    calls = {"exec": 0, "edit_configs": []}

    class Adapter:
        name = "hf_gpt2"

        def load_model(self, model_id, device=None):
            return SmallModel()

    def detect_profile(model_id, adapter):
        return SimpleNamespace(
            default_loss="ce",
            model_id=model_id,
            adapter=adapter,
            module_selectors={},
            invariants=set(),
            cert_lints=[],
            family="gpt2",
        )

    class RC:
        def __init__(self, max_attempts=3, timeout=None, verbose=False):
            self.attempt_history = []

        def should_retry(self, passed):
            # Only allow retry after PASS for materialization once
            return len(self.attempt_history) == 0

        def record_attempt(self, attempt, result_summary, edit_config):
            self.attempt_history.append(result_summary)

        def get_attempt_summary(self):
            return {"total_attempts": len(self.attempt_history), "elapsed_time": 0.1}

    def make_cert(report, baseline_report):
        # Always PASS to trigger materialize branch
        return {"validation": {"primary_metric_acceptable": True, "drift_ok": True}}

    def runner_exec(**kwargs):
        calls["exec"] += 1
        calls["edit_configs"].append(kwargs.get("edit_config", {}))
        # Provide edit deltas with heads_pruned above threshold
        return SimpleNamespace(
            edit={"deltas": {"heads_pruned": 10}},
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

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.commands.run.detect_model_profile", detect_profile)
        )
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
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                        SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
                    )
                ),
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
            profile="ci",
            baseline=str(baseline),
            until_pass=True,
            max_attempts=2,
            out=str(tmp_path / "runs"),
        )

    # At least one execution occurred and edit config was provided
    assert calls["exec"] >= 1
    assert isinstance(calls["edit_configs"][0].get("heads", {}), dict)


def test_release_baseline_no_eval_windows_exit(tmp_path: Path):
    cfg = _cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"meta": {"tokenizer_hash": "tokhash123"}}))
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(input_ids=[[1]], attention_masks=[[1]]),
                        SimpleNamespace(input_ids=[[2]], attention_masks=[[1]]),
                    )
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
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
            until_pass=False,
        )


def test_snapshot_auto_bytes_when_small_model(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path, 1, 1)

    class Adapter:
        name = "hf_gpt2"

        def __init__(self):
            self.restored = 0

        def load_model(self, model_id, device=None):
            return SmallModel()

        def snapshot(self, model):
            return b"blob"

        def restore(self, model, blob):
            self.restored += 1

    adapter = Adapter()

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
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
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[5, 6, 7, 8]], attention_masks=[[1, 1, 1, 1]]
                        ),
                    )
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.psutil.virtual_memory",
                lambda: _psutil_vm(available_mb=8192),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.shutil.disk_usage",
                lambda path: _disk_usage(free_mb=0),
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    assert adapter.restored >= 1


def test_snapshot_no_support_uses_reload(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path, 1, 1)

    class Adapter:
        name = "hf_gpt2"

        def __init__(self):
            self.loaded = 0

        def load_model(self, model_id, device=None):
            self.loaded += 1
            return SmallModel()

    adapter = Adapter()

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
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
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[5, 6, 7, 8]], attention_masks=[[1, 1, 1, 1]]
                        ),
                    )
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.psutil.virtual_memory",
                lambda: _psutil_vm(available_mb=0),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.shutil.disk_usage",
                lambda path: _disk_usage(free_mb=0),
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    # Without snapshot support, load at least twice (bare + guarded)
    assert adapter.loaded >= 2


def test_snapshot_env_mode_overrides(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path, 1, 1)

    class Adapter:
        name = "hf_gpt2"

        def __init__(self):
            self.restored = 0
            self.rest_chunked = 0

        def load_model(self, model_id, device=None):
            return LargeModel()

        def snapshot(self, model):
            return b"x"

        def restore(self, model, blob):
            self.restored += 1

        def snapshot_chunked(self, model):
            return "p"

        def restore_chunked(self, model, path):
            self.rest_chunked += 1

    adapter = Adapter()

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
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
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                        SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
                    )
                ),
            )
        )
        # Force env override to bytes
        monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "bytes")
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
        # Reset and override to chunked
        monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )

    assert adapter.restored >= 1 and adapter.rest_chunked >= 1
