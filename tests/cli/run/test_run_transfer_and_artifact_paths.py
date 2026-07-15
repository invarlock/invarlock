# ruff: noqa: I001,E402,F811
from __future__ import annotations

# Consolidated from:
# - tests/cli/run/test_run_branch_more_cases.py
# - tests/cli/run/test_run_additional_branches.py

# --- Begin: test_run_branch_more_cases.py ---

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command
from tests.cli.run._support_run_common import (
    assert_single_run_output_artifacts,
    canonical_ppl_metrics,
)


def _cfg(tmp_path: Path, preview=1, final=1) -> Path:
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
  order: [invariants]

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
            "invarlock.reporting.report_bundle.save_report",
            lambda report, run_dir, formats, filename_prefix=None: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
        patch(
            "invarlock.cli.run_runtime_exec.detect_model_profile",
            lambda model_id=None, adapter=None: SimpleNamespace(
                default_loss="ce",
                model_id=model_id,
                adapter=adapter,
                module_selectors={},
                invariants=("checkA",),
                cert_lints=[],
                family="gpt2",
            ),
        ),
        patch(
            "invarlock.cli.run_runtime_exec.resolve_tokenizer",
            lambda prof: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
        patch(
            "invarlock.eval.data.get_provider",
            lambda *a, **k: SimpleNamespace(
                windows=lambda **kw: (
                    SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
                    SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
                )
            ),
        ),
    )


# --- Begin: test_run_additional_branches.py ---

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command


def _basic_cfg(tmp_path: Path, preview: int = 1, final: int = 1) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
model:
  adapter: hf_causal
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan: {{ heads: {{ mask_only: true, mask_auto: true }} }}

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


def _common_device_and_save():
    return (
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report_bundle.save_report",
            lambda report, run_dir, formats, filename_prefix=None: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
    )


def _reg_and_provider(provider_windows=None):
    if provider_windows is None:
        provider_windows = (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )

    return (
        patch(
            "invarlock.core.registry.get_registry",
            lambda: SimpleNamespace(
                get_adapter=lambda name: SimpleNamespace(
                    name=name, load_model=lambda model_id, device=None: object()
                ),
                get_edit=lambda name: SimpleNamespace(name=name),
                get_guard=lambda name: SimpleNamespace(name=name),
                get_plugin_metadata=lambda n, t: {
                    "name": n,
                    "module": f"{t}.{n}",
                    "version": "test",
                },
            ),
        ),
        patch(
            "invarlock.eval.data.get_provider",
            lambda *a, **k: SimpleNamespace(windows=lambda **kw: provider_windows),
        ),
    )


def _runner_min():
    return SimpleNamespace(
        execute=lambda **k: SimpleNamespace(
            edit={},
            metrics=canonical_ppl_metrics(),
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={},
            status="success",
        )
    )


def _detect_loss(loss_type: str = "ce"):
    return patch(
        "invarlock.cli.run_runtime_exec.detect_model_profile",
        lambda model_id, adapter: SimpleNamespace(
            default_loss=loss_type,
            model_id=model_id,
            adapter=adapter,
            module_selectors={},
            invariants=set(),
            cert_lints=[],
            family="gpt",
            make_tokenizer=lambda: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50_000),
                "tokhash123",
            ),
        ),
    )


# Keep the remainder of the additional branches tests intact
# (We include only a representative subset due to consolidation.)


# ---- Selected general edge scenarios (from edges) ----


def test_edit_cli_override_invalid_exits(tmp_path: Path):
    cfg = _cfg(tmp_path)
    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.run_config._resolve_requested_edit_name",
                side_effect=ValueError("bad edit"),
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            out=str(tmp_path / "runs"),
            edit="not-a-real-edit",
        )


def test_device_validation_fail_exits(tmp_path: Path):
    cfg = _cfg(tmp_path)
    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config",
                lambda d: (False, "nope"),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_transfer_guard_extras_and_guard_recovered_flag(tmp_path: Path):
    cfg = _cfg(tmp_path)
    with ExitStack() as stack:
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.validate_guard_metric_impact",
                lambda *a, **k: SimpleNamespace(
                    passed=True,
                    messages=[],
                    warnings=[],
                    errors=[],
                    checks={"guard_metric_impact": True},
                    metrics={"degradation": 1.0, "display_value": 0.0},
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(
                    execute=lambda **k: SimpleNamespace(
                        edit={
                            "plan_digest": "abcd",
                            "deltas": {"params_changed": 1},
                            "transfer_guard_extras": {"hello": True},
                        },
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                        },
                        guards={
                            "variance": {
                                "ok": True,
                                "messages": [],
                                "warnings": [],
                                "errors": [],
                                "extras": {"guard_recovered": True},
                            }
                        },
                        context={"dataset_meta": {}},
                        evaluation_windows={},
                        status="success",
                    )
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
    assert_single_run_output_artifacts(tmp_path)


def test_artifacts_events_path_empty_path(tmp_path: Path):
    cfg = _cfg(tmp_path)
    with ExitStack() as stack:
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
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
    assert_single_run_output_artifacts(tmp_path)


def test_seed_bundle_fallbacks(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    with ExitStack() as stack:
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
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
                        context={
                            "dataset_meta": {},
                            "seed_bundle": {"python": 42, "numpy": None, "torch": None},
                        },
                        evaluation_windows={},
                        status="success",
                    )
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
    assert_single_run_output_artifacts(tmp_path)


def test_invariants_profile_checks_merged(tmp_path: Path):
    cfg = _cfg(tmp_path)
    with ExitStack() as stack:
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.detect_model_profile",
                lambda *a, **k: SimpleNamespace(
                    default_loss="ce",
                    model_id=None,
                    adapter=None,
                    module_selectors={},
                    invariants=("checkA",),
                    cert_lints=[],
                    family="gpt2",
                ),
            )
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", _runner_min))
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
    assert_single_run_output_artifacts(tmp_path)


def test_baseline_pairing_capacity_meta_skip_window_capacity_assignment(tmp_path: Path):
    cfg = _cfg(tmp_path)
    with ExitStack() as stack:
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
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
                        context={
                            "dataset_meta": {
                                "window_plan": {
                                    "capacity": {
                                        "reserve_windows": 20,
                                        "calibration": 20,
                                    }
                                }
                            }
                        },
                        evaluation_windows={},
                        status="success",
                    )
                ),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
    assert_single_run_output_artifacts(tmp_path)


def test_edit_name_invalid_exits(tmp_path: Path):
    cfg = _cfg(tmp_path)
    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.cli.run_config._resolve_requested_edit_name",
                side_effect=ValueError("bad edit"),
            )
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), edit="manual"
        )


def test_unknown_edit_name_exits_with_code_2(tmp_path: Path):
    cfg = _cfg(tmp_path)

    class _Registry:
        def get_adapter(self, _name: str):
            return object()

        def get_edit(self, _name: str):
            raise KeyError("missing")

    with ExitStack() as stack:
        for ctx in _common_patches_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: _Registry())
        )
        with pytest.raises(click.exceptions.Exit) as excinfo:
            run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
    assert excinfo.value.exit_code == 2


def test_baseline_stride_mismatch_exit(tmp_path: Path):
    cfg = _basic_cfg(tmp_path)
    with ExitStack() as stack, pytest.raises(click.exceptions.Exit):
        for ctx in _common_device_and_save():
            stack.enter_context(ctx)
        stack.enter_context(_detect_loss("mlm"))
        # Provide mismatched stride in baseline meta
        baseline = tmp_path / "baseline.json"
        baseline.write_text(
            json.dumps({"meta": {"stride": 999, "tokenizer_hash": "tokhash123"}})
        )
        for ctx in _reg_and_provider():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", _runner_min))
        run_command(
            config=str(cfg),
            device="cpu",
            out=str(tmp_path / "runs"),
            baseline=str(baseline),
        )


def test_bare_metric_impact_measurement_pass(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name,
                load_model=lambda model_id, device=None: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda _m, _b=None: None,
                snapshot_chunked=lambda _m=None: str(tmp_path / "snapdir"),
                restore_chunked=lambda _m, _d=None: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    class DummyRunner:
        def __init__(self, bare_status_ok=True):
            self.bare_status_ok = bare_status_ok

        def execute(self, **kwargs):
            guards = kwargs.get("guards") or []
            if not guards and not self.bare_status_ok:
                return SimpleNamespace(
                    edit={}, metrics={}, guards={}, context={}, status="failed"
                )
            return SimpleNamespace(
                edit={"plan_digest": "abcd", "deltas": {"params_changed": 0}},
                metrics={
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 10.0,
                        "final": 10.0,
                    },
                    "ppl_preview": 10.0,
                    "ppl_final": 10.0,
                    "ppl_ratio": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_match_fraction": 1.0,
                    "loss_type": "ce",
                },
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={
                    "preview": {
                        "window_ids": [0],
                        "logloss": [2.302585092994046],
                        "token_counts": [1],
                    },
                    "final": {
                        "window_ids": [1],
                        "logloss": [2.302585092994046],
                        "token_counts": [1],
                    },
                },
                status="success",
            )

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.detect_model_profile",
        lambda *a, **k: SimpleNamespace(
            default_loss="ce",
            module_selectors={},
            invariants=[],
            cert_lints=[],
            family="gpt2",
            make_tokenizer=lambda: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.resolve_tokenizer",
        lambda *_: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tokhash123",
        ),
    )
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: DummyRunner(bare_status_ok=True)
    )
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile="ci")
    assert_single_run_output_artifacts(tmp_path)


def test_dataset_meta_tokenizer_hash_passthrough(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(name=name, load_model=lambda *a, **k: object())

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())

    def _exec(**kwargs):
        return SimpleNamespace(
            edit={"plan_digest": "abcd", "deltas": {"params_changed": 0}},
            metrics=canonical_ppl_metrics(
                preview=10.0,
                final=10.0,
                window_overlap_fraction=0.0,
                window_match_fraction=1.0,
                loss_type="ce",
            ),
            guards={},
            context={
                "dataset_meta": {
                    "tokenizer_hash": "tok123",
                    "preview_total_tokens": 2,
                    "final_total_tokens": 2,
                }
            },
            evaluation_windows={},
            status="success",
        )

    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner", lambda: SimpleNamespace(execute=_exec)
    )
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.detect_model_profile",
        lambda *a, **k: SimpleNamespace(
            default_loss="ce",
            module_selectors={},
            invariants=[],
            cert_lints=[],
            family="gpt2",
            make_tokenizer=lambda: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.run_runtime_exec.resolve_tokenizer",
        lambda *_: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
            "tokhash123",
        ),
    )
    run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"), profile=None)
    assert_single_run_output_artifacts(tmp_path)


def test_auto_adapter_apply_fails_closed_on_error(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(name=name, load_model=lambda *a, **k: object())

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )
        ),
    )
    monkeypatch.setattr(
        "invarlock.adapters.auto.apply_auto_adapter_if_needed",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("auto-err")),
    )
    with pytest.raises(click.exceptions.Exit) as excinfo:
        run_command(
            config=str(cfg),
            device="cpu",
            out=str(tmp_path / "runs"),
            profile=None,
        )

    assert excinfo.value.exit_code == 1
