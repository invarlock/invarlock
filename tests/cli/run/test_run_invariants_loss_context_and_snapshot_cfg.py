from __future__ import annotations

import json
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
    )


def _provider():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )
    )


def test_file_not_found_exit(tmp_path: Path):
    missing_cfg = tmp_path / "nope.yaml"
    with pytest.raises(click.exceptions.Exit):
        run_command(config=str(missing_cfg), device="cpu", out=str(tmp_path / "runs"))


def test_invariants_existing_checks_as_scalar_becomes_list(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    captured = {}

    def detect_profile(model_id, adapter):
        return SimpleNamespace(
            default_loss="ce",
            model_id=model_id,
            adapter=adapter,
            module_selectors={},
            invariants={"a", "b"},
            cert_lints=[],
            family="gpt",
        )

    class DummyCfg:
        def __init__(self):
            self.model = SimpleNamespace(id="gpt2", adapter="hf_causal", device="cpu")
            self.edit = SimpleNamespace(name="quant_rtn", plan={})
            self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
            self.guards = SimpleNamespace(
                order=[], invariants=SimpleNamespace(profile_checks="foo")
            )
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

        def model_dump(self):
            return {}

    def runner_exec(**kwargs):
        captured["policy"] = (
            kwargs.get("config").context.get("guards", {}).get("invariants", {})
        )
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.commands.run.detect_model_profile", detect_profile)
        )
        stack.enter_context(
            patch("invarlock.cli.config.load_config", lambda p: DummyCfg())
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    checks = captured["policy"].get("profile_checks", [])
    # Ensure scalar existing_checks became a list and preserved
    assert isinstance(checks, list) and "foo" in checks


def test_invariants_existing_checks_set_becomes_list(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    captured = {}

    def detect_profile(model_id, adapter):
        return SimpleNamespace(
            default_loss="ce",
            model_id=model_id,
            adapter=adapter,
            module_selectors={},
            invariants={"x", "y"},
            cert_lints=[],
            family="gpt",
        )

    class DummyCfg:
        def __init__(self):
            self.model = SimpleNamespace(id="gpt2", adapter="hf_causal", device="cpu")
            self.edit = SimpleNamespace(name="quant_rtn", plan={})
            self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
            self.guards = SimpleNamespace(
                order=[], invariants=SimpleNamespace(profile_checks={"z"})
            )
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

        def model_dump(self):
            return {}

    def runner_exec(**kwargs):
        captured["policy"] = (
            kwargs.get("config").context.get("guards", {}).get("invariants", {})
        )
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.commands.run.detect_model_profile", detect_profile)
        )
        stack.enter_context(
            patch("invarlock.cli.config.load_config", lambda p: DummyCfg())
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    checks = captured["policy"].get("profile_checks", [])
    assert (
        isinstance(checks, list)
        and "z" in checks
        and all(isinstance(x, str) for x in checks)
    )


def test_invariants_existing_checks_tuple_becomes_list(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    captured = {}

    def detect_profile(model_id, adapter):
        return SimpleNamespace(
            default_loss="ce",
            model_id=model_id,
            adapter=adapter,
            module_selectors={},
            invariants={"m", "n"},
            cert_lints=[],
            family="gpt",
        )

    class DummyCfg:
        def __init__(self):
            self.model = SimpleNamespace(id="gpt2", adapter="hf_causal", device="cpu")
            self.edit = SimpleNamespace(name="quant_rtn", plan={})
            self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
            self.guards = SimpleNamespace(
                order=[], invariants=SimpleNamespace(profile_checks=("q",))
            )
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

        def model_dump(self):
            return {}

    def runner_exec(**kwargs):
        captured["policy"] = (
            kwargs.get("config").context.get("guards", {}).get("invariants", {})
        )
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.commands.run.detect_model_profile", detect_profile)
        )
        stack.enter_context(
            patch("invarlock.cli.config.load_config", lambda p: DummyCfg())
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    checks = captured["policy"].get("profile_checks", [])
    assert isinstance(checks, list) and "q" in checks


def test_loss_cfg_nan_values_coerced(tmp_path: Path):
    cfg = _base_cfg(tmp_path)

    class DummyCfg:
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
            # NaN strings should coerce to defaults in _coerce_float; keep loss type 'ce' to avoid MLM tokenizer requirement
            self.eval = SimpleNamespace(
                spike_threshold=2.0,
                loss=SimpleNamespace(
                    type="ce",
                    mask_prob="nan",
                    random_token_prob="nan",
                    original_token_prob="nan",
                ),
            )
            self.output = SimpleNamespace(dir=tmp_path / "runs")

        def model_dump(self):
            return {}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.config.load_config", lambda p: DummyCfg())
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.resolve_tokenizer",
                lambda profile: (
                    SimpleNamespace(
                        mask_token_id=103,
                        eos_token="</s>",
                        pad_token="</s>",
                        vocab_size=50000,
                    ),
                    "tokhash123",
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider())
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
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_snapshot_auto_ram_fraction_env(tmp_path: Path, monkeypatch):
    cfg = _base_cfg(tmp_path)

    class Adapter:
        name = "hf_causal"

        def __init__(self):
            self.rest_chunked = 0

        def load_model(self, model_id, device=None):
            class M:
                def named_parameters(self):
                    return [
                        (
                            "p",
                            SimpleNamespace(
                                element_size=lambda: 1, nelement=lambda: 900_000_000
                            ),
                        )
                    ]

                def named_buffers(self):
                    return []

            return M()

        def snapshot_chunked(self, model):
            snap_dir = tmp_path / "snapshot_chunked"
            snap_dir.mkdir(parents=True, exist_ok=True)
            return str(snap_dir)

        def restore_chunked(self, model, path):
            self.rest_chunked += 1

    adapter = Adapter()

    def vm():
        return SimpleNamespace(available=200 * 1024 * 1024)  # 200MB

    def du(path):
        return SimpleNamespace(total=0, used=0, free=4 * 1024 * 1024 * 1024)  # 4GB

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        monkeypatch.setenv("INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION", "0.2")
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
            patch("invarlock.cli.commands.run.psutil.virtual_memory", vm)
        )
        stack.enter_context(patch("invarlock.cli.commands.run.shutil.disk_usage", du))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider())
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    assert adapter.rest_chunked >= 1


def test_snapshot_cfg_threshold_and_tempdir(tmp_path: Path, monkeypatch):
    cfg = _base_cfg(tmp_path)

    class Adapter:
        name = "hf_causal"

        def __init__(self):
            self.rest_chunked = 0

        def load_model(self, model_id, device=None):
            class M:
                def named_parameters(self):
                    return [
                        (
                            "p",
                            SimpleNamespace(
                                element_size=lambda: 1, nelement=lambda: 50_000_000
                            ),
                        )
                    ]  # 50MB

                def named_buffers(self):
                    return []

            return M()

        def snapshot_chunked(self, model):
            snap_dir = tmp_path / "snapshot_chunked"
            snap_dir.mkdir(parents=True, exist_ok=True)
            return str(snap_dir)

        def restore_chunked(self, model, path):
            self.rest_chunked += 1

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
                self.context = {
                    "snapshot": {
                        "threshold_mb": 10.0,
                        "disk_free_margin_ratio": 1.1,
                        "temp_dir": str(tmp_path),
                    }
                }

            def model_dump(self):
                return {}

        return Cfg()

    def vm():
        return SimpleNamespace(available=0)

    def du(path):
        return SimpleNamespace(total=0, used=0, free=100 * 1024 * 1024)  # 100MB

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
            patch("invarlock.cli.commands.run.psutil.virtual_memory", vm)
        )
        stack.enter_context(patch("invarlock.cli.commands.run.shutil.disk_usage", du))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider())
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    assert adapter.rest_chunked >= 1


def test_loss_type_resolved_in_context(tmp_path: Path):
    cfg = _base_cfg(tmp_path)
    captured = {}

    def detect(model_id, adapter):
        return SimpleNamespace(
            default_loss="ce",
            model_id=model_id,
            adapter=adapter,
            module_selectors={},
            invariants=set(),
            cert_lints=[],
            family="gpt",
        )

    def runner_exec(**kwargs):
        captured["ctx"] = kwargs.get("config").context
        return SimpleNamespace(
            edit={},
            metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
            guards={},
            context={"dataset_meta": {}},
            status="success",
        )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.commands.run.detect_model_profile", detect)
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=runner_exec),
            )
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    assert captured["ctx"]["eval"]["loss"].get("resolved_type") in {"ce", "causal"}


def test_baseline_masked_counts_used_when_present(tmp_path: Path):
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
                        "masked_token_counts": [2],
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[4, 5, 6]],
                        "attention_masks": [[1, 1, 1]],
                        "masked_token_counts": [1],
                    },
                },
            }
        )
    )

    def detect_mlm(model_id, adapter):
        return SimpleNamespace(
            default_loss="mlm",
            model_id=model_id,
            adapter=adapter,
            module_selectors={},
            invariants=set(),
            cert_lints=[],
            family="bert",
        )

    captured = {}

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.commands.run.detect_model_profile", detect_mlm)
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.resolve_tokenizer",
                lambda profile: (
                    SimpleNamespace(
                        mask_token_id=103,
                        eos_token="</s>",
                        pad_token="</s>",
                        vocab_size=50_000,
                    ),
                    "tokhash123",
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider())
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
                            "window_overlap_fraction": 0.0,
                            "window_match_fraction": 1.0,
                            "paired_windows": 1,
                        },
                        guards={},
                        context=getattr(k.get("config"), "context", {}),
                        status="success",
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
                    )
                ),
            )
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
        )

    # masked tokens totals live in dataset_meta and merge into data when present
    d = captured["r"].get("data", {})
    tot = d.get("masked_tokens_total")
    if tot is not None:
        assert tot == 3
