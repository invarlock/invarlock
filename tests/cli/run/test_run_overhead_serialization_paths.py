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
            "invarlock.reporting.report_files.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
        patch(
            "invarlock.cli.run_runtime.resolve_tokenizer",
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


def _runner_success():
    return SimpleNamespace(
        execute=lambda **k: SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
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


# --- Selected serialization/OptionInfo/psutil/env fallback edge tests ---


def test_overhead_threshold_bad_type_uses_default(tmp_path: Path):
    # overhead_threshold not a float → fallback to GUARD_OVERHEAD_THRESHOLD internally, no crash
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
            checks={},
            metrics={"overhead_percent": 0.5, "overhead_threshold": "bad"},
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        for target in (
            "invarlock.reporting.validate.validate_guard_overhead",
            "invarlock.cli.run_runtime.validate_guard_overhead",
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
                "invarlock.cli.run_runtime.psutil.virtual_memory",
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


def test_save_report_failure_bubbles_to_exit(tmp_path: Path):
    cfg = _base_cfg(tmp_path)

    def bad_save(*args, **kwargs):
        raise RuntimeError("cannot save")

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.reporting.report_files.save_report", bad_save)
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
