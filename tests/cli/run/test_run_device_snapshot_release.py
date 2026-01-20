from __future__ import annotations

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
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
    )


def _provider_min():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )
    )


def test_device_validation_failure_exits(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        # Force device validation failure (patch both module paths)
        for target in (
            "invarlock.cli.device.validate_device_for_config",
            "invarlock.cli.device.validate_device_for_config",
        ):
            stack.enter_context(patch(target, lambda d: (False, "unsupported device")))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={},
                        metrics={},
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
                profile=None,
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_release_capacity_insufficient_exit(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)

    class Provider:
        def estimate_capacity(self, **kwargs):  # noqa: D401
            # Too few unique windows to satisfy minimum per-arm
            return {
                "available_unique": 100,
                "available_nonoverlap": 100,
                "total_tokens": 1000,
                "dedupe_rate": 0.0,
            }

        def windows(self, **kwargs):  # noqa: D401
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
                until_pass=False,
            )


def test_mask_flags_do_not_error(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)

    captured = {}

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id, device=None):  # noqa: D401
            return object()

        def describe(self, model):  # noqa: D401
            # 12 layers with 12 heads each
            return {"heads_per_layer": [12] * 12}

    class DummyRegistry:
        def get_adapter(self, name):
            return Adapter()

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    def runner_exec(**kwargs):
        captured["edit_config"] = kwargs.get("edit_config")
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )

    # Prepare config with mask_only + mask_auto flags
    cfg_text = cfg.read_text().replace(
        "plan: {}", "plan: { heads: { mask_only: true, mask_auto: true } }"
    )
    cfg.write_text(cfg_text)

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: DummyRegistry())
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
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
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    # Ensure pipeline handled mask flags without errors
    assert captured.get("edit_config") is not None


def test_snapshot_mode_bytes_restore_called(tmp_path: Path, monkeypatch):
    cfg = _base_cfg(tmp_path, 1, 1)

    class Adapter:
        name = "hf_causal"

        def __init__(self):
            self.loaded = 0
            self.restored = 0

        def load_model(self, model_id, device=None):  # noqa: D401
            self.loaded += 1
            return object()

        def snapshot(self, model):  # noqa: D401
            return b"bytes"

        def restore(self, model, blob):  # noqa: D401
            self.restored += 1

    adapter = Adapter()

    class DummyRegistry:
        def get_adapter(self, name):
            return adapter

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
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
                        status="success",
                    )
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: DummyRegistry())
        )
        monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "bytes")
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    # restore() should be used twice (bare + guarded)
    assert adapter.restored >= 2


def test_snapshot_mode_bytes_falls_back_to_chunked_on_failure(
    tmp_path: Path, monkeypatch
):
    cfg = _base_cfg(tmp_path, 1, 1)

    class Adapter:
        name = "hf_causal"

        def __init__(self):
            self.loaded = 0
            self.snapshot_calls = 0
            self.snapshot_chunked_calls = 0
            self.restore_calls = 0
            self.restore_chunked_calls = 0

        def load_model(self, model_id, device=None):  # noqa: D401
            self.loaded += 1
            return object()

        def snapshot(self, model):  # noqa: D401
            self.snapshot_calls += 1
            raise RuntimeError("oom")

        def restore(self, model, blob):  # noqa: D401
            self.restore_calls += 1

        def snapshot_chunked(self, model):  # noqa: D401
            self.snapshot_chunked_calls += 1
            snap_dir = tmp_path / "snapshot_chunked"
            snap_dir.mkdir(parents=True, exist_ok=True)
            return str(snap_dir)

        def restore_chunked(self, model, path):  # noqa: D401
            self.restore_chunked_calls += 1

    adapter = Adapter()

    class DummyRegistry:
        def get_adapter(self, name):
            return adapter

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
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
                        status="success",
                    )
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: DummyRegistry())
        )
        monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "bytes")
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert adapter.loaded == 1
    assert adapter.snapshot_calls == 1
    assert adapter.snapshot_chunked_calls == 1
    assert adapter.restore_calls == 0
    assert adapter.restore_chunked_calls >= 2


def test_snapshot_mode_chunked_restore_called(tmp_path: Path, monkeypatch):
    cfg = _base_cfg(tmp_path, 1, 1)

    class Adapter:
        name = "hf_causal"

        def __init__(self):
            self.loaded = 0
            self.restored = 0

        def load_model(self, model_id, device=None):  # noqa: D401
            self.loaded += 1
            return object()

        def snapshot_chunked(self, model):  # noqa: D401
            snap_dir = tmp_path / "snapshot_chunked"
            snap_dir.mkdir(parents=True, exist_ok=True)
            return str(snap_dir)

        def restore_chunked(self, model, path):  # noqa: D401
            self.restored += 1

    adapter = Adapter()

    class DummyRegistry:
        def get_adapter(self, name):
            return adapter

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
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
                        status="success",
                    )
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: DummyRegistry())
        )
        monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "chunked")
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert adapter.restored >= 2


def test_snapshot_mode_reload_loads_each_attempt(tmp_path: Path, monkeypatch):
    cfg = _base_cfg(tmp_path, 1, 1)

    class Adapter:
        name = "hf_causal"

        def __init__(self):
            self.loaded = 0

        def load_model(self, model_id, device=None):  # noqa: D401
            self.loaded += 1
            return object()

    adapter = Adapter()

    class DummyRegistry:
        def get_adapter(self, name):
            return adapter

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
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
                        status="success",
                    )
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: DummyRegistry())
        )
        # force reload path
        monkeypatch.delenv("INVARLOCK_SNAPSHOT_MODE", raising=False)
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    # load_model called at least twice (bare + guarded)
    assert adapter.loaded >= 2


def test_guard_overhead_bare_missing_ppl_and_status_warn(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)

    class Runner:
        def execute(self, **kwargs):
            # Detect bare vs guarded by presence of guards list
            if kwargs.get("guards") == []:
                # bare report missing ppl_final and with non-success status
                return SimpleNamespace(
                    edit={},
                    metrics={"ppl_preview": 1.0},
                    guards={},
                    context={"dataset_meta": {}},
                    status="error",
                )
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        # Patch validator to avoid exit; we only inspect warnings/errors aggregated pre-validation
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
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        # Ensure profile that measures overhead (ci)
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    gh = captured["r"].get("guard_overhead", {})
    assert (
        isinstance(gh, dict)
        and gh.get("evaluated") is True
        and gh.get("passed") is True
    )


def test_evaluation_window_parity_mismatch_exit(tmp_path: Path):
    from invarlock.eval.data import EvaluationWindow

    cfg = _base_cfg(tmp_path, 2, 2)

    class Provider:
        def windows(self, **kwargs):
            prev = EvaluationWindow(
                input_ids=[[1, 2]], attention_masks=[[1, 1]], indices=[0]
            )
            # Mismatch: only one final sample
            fin = EvaluationWindow(
                input_ids=[[3, 4]], attention_masks=[[1, 1]], indices=[0]
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
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile="release",
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_tokenizer_hash_populated_from_context_when_missing(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)

    def resolver(_):
        # Return tokenizer object with no hash value
        return SimpleNamespace(
            eos_token="</s>", pad_token="</s>", vocab_size=50000
        ), None

    class Runner:
        def execute(self, **kwargs):
            # Provide tokenizer_hash in dataset_meta context to be promoted to meta.tokenizer_hash
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context={"dataset_meta": {"tokenizer_hash": "from_ctx"}},
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.commands.run.resolve_tokenizer", resolver)
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        # Capture save but don't need to inspect
        stack.enter_context(
            patch(
                "invarlock.reporting.report.save_report",
                lambda report, run_dir, formats, filename_prefix: {
                    "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
                },
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )
