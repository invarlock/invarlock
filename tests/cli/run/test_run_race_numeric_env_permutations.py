from __future__ import annotations

import math
from contextlib import ExitStack
from itertools import permutations
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
  adapter: hf_gpt2
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


def _provider_min():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )
    )


def _common_ce_detect_ce():
    return (
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
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
            lambda profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
        patch(
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
    )


def test_output_dir_deleted_before_save_report(tmp_path: Path):
    # Simulate run_dir deleted just before saving report -> triggers final exception path
    cfg = _base_cfg(tmp_path)
    called = {"once": False}

    def cap_save(report, run_dir, formats, filename_prefix=None):
        # Delete the run_dir right before attempting to return paths
        if not called["once"]:
            called["once"] = True
            try:
                from shutil import rmtree

                rmtree(run_dir, ignore_errors=True)
            except Exception:
                pass
        raise FileNotFoundError("run_dir vanished")

    with ExitStack() as stack:
        for ctx in _common_ce_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
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
        with pytest.raises(click.exceptions.Exit):
            run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_permission_error_on_run_dir_creation(tmp_path: Path):
    # Raise PermissionError when creating the second directory (run_dir)
    cfg = _base_cfg(tmp_path)
    mkdir_calls = {"count": 0}

    def guarded_mkdir(self, parents=False, exist_ok=False):  # noqa: D401
        mkdir_calls["count"] += 1
        # First mkdir is for output_dir; second is for run_dir
        if mkdir_calls["count"] == 2:
            raise PermissionError("read-only parent")
        return None

    with ExitStack() as stack:
        for ctx in _common_ce_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("pathlib.Path.mkdir", guarded_mkdir))
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
            run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_memoryerror_on_model_load(tmp_path: Path):
    # MemoryError injection on adapter.load_model should be caught by outer exception path
    cfg = _base_cfg(tmp_path)

    class Adapter:
        name = "hf_gpt2"

        def load_model(self, model_id, device=None):
            raise MemoryError("OOM")

    with ExitStack() as stack:
        for ctx in _common_ce_detect_ce():
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
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_vars_failure_in_to_serialisable_dict(tmp_path: Path):
    # Use a context object with __slots__ to make vars() raise TypeError
    cfg = _base_cfg(tmp_path)

    class NoVars:
        __slots__ = ("a",)

        def __init__(self):
            self.a = 1

    class Cfg:
        def __init__(self):
            self.model = SimpleNamespace(id="gpt2", adapter="hf_gpt2", device="cpu")
            self.edit = SimpleNamespace(name="structured", plan={})
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
            self.context = NoVars()

        def model_dump(self):
            return {}

    with ExitStack() as stack:
        for ctx in _common_ce_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.cli.config.load_config", lambda p: Cfg()))
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
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_invalid_file_encoding_baseline(tmp_path: Path):
    # Baseline with invalid UTF-8 to trigger UnicodeDecodeError path for baseline load
    cfg = _base_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(b"\xff\xfe\xfa\xfb")  # invalid utf-8

    with ExitStack() as stack:
        for ctx in _common_ce_detect_ce():
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
        # Should not crash; falls back to dataset schedule
        run_command(
            config=str(cfg),
            device="cpu",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
        )


def test_env_var_poisoning_for_tmpdir_and_debug(tmp_path: Path, monkeypatch):
    # TMPDIR empty -> snapshot chooser falls back to /tmp; INVARLOCK_DEBUG_TRACE odd chars -> no crash
    cfg = _base_cfg(tmp_path)

    class Adapter:
        name = "hf_gpt2"

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

    with ExitStack() as stack:
        for ctx in _common_ce_detect_ce():
            stack.enter_context(ctx)
        monkeypatch.setenv("TMPDIR", "")
        # Avoid embedded null; use unicode-only string
        monkeypatch.setenv("INVARLOCK_DEBUG_TRACE", "\u2603 odd")
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
                lambda: SimpleNamespace(available=200 * 1024 * 1024),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.shutil.disk_usage",
                lambda path: SimpleNamespace(
                    total=0, used=0, free=4 * 1024 * 1024 * 1024
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_debug_trace_with_mlm_masks_prints(tmp_path: Path, monkeypatch):
    # Enable INVARLOCK_DEBUG_TRACE and run MLM baseline to exercise debug print branches
    cfg = _base_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        """
{
  "meta": {"tokenizer_hash": "tokhash123"},
  "evaluation_windows": {
    "preview": {"window_ids": [0], "input_ids": [[1,2,3]], "attention_masks": [[1,1,1]]},
    "final": {"window_ids": [1], "input_ids": [[4,5,6]], "attention_masks": [[1,1,1]]}
  }
}
        """.strip()
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

    with ExitStack() as stack:
        monkeypatch.setenv("INVARLOCK_DEBUG_TRACE", "1")
        stack.enter_context(patch("invarlock.cli.device.resolve_device", lambda d: d))
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
        stack.enter_context(
            patch("invarlock.cli.commands.run.detect_model_profile", detect_mlm)
        )
        for target in (
            "invarlock.cli.commands.run.resolve_tokenizer",
            "invarlock.cli.commands.run.resolve_tokenizer",
        ):
            stack.enter_context(
                patch(
                    target,
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
        run_command(
            config=str(cfg),
            device="cpu",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
        )


def test_nan_inf_metrics_propagation(tmp_path: Path):
    # Runner returns NaN/inf metrics; drift print + report should not crash
    cfg = _base_cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={
                    "ppl_preview": math.inf,
                    "ppl_final": math.inf,
                    "ppl_ratio": math.nan,
                    "window_overlap_fraction": 0.0,
                    "window_match_fraction": 1.0,
                },
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_ce_detect_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))


def test_mlm_probability_inversion(tmp_path: Path):
    # Configure MLM with random_token_prob + original_token_prob > 1.0 → negative replace_threshold branch
    cfg = _base_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        """
{
  "meta": {"tokenizer_hash": "tokhash123"},
  "evaluation_windows": {
    "preview": {"window_ids": [0], "input_ids": [[1,2,3]], "attention_masks": [[1,1,1]]},
    "final": {"window_ids": [1], "input_ids": [[4,5,6]], "attention_masks": [[1,1,1]]}
  }
}
        """.strip()
    )

    class Cfg:
        def __init__(self):
            self.model = SimpleNamespace(adapter="hf_gpt2", id="gpt2", device="cpu")
            self.edit = SimpleNamespace(name="structured", plan={})
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
            self.guards = SimpleNamespace(order=[])
            self.eval = SimpleNamespace(
                spike_threshold=2.0,
                loss=SimpleNamespace(
                    type="mlm",
                    mask_prob=0.5,
                    random_token_prob=0.8,
                    original_token_prob=0.3,
                ),
            )
            self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
            self.output = SimpleNamespace(dir=tmp_path / "runs")

        def model_dump(self):
            return {}

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

    with ExitStack() as stack:
        stack.enter_context(patch("invarlock.cli.device.resolve_device", lambda d: d))
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
        stack.enter_context(patch("invarlock.cli.config.load_config", lambda p: Cfg()))
        stack.enter_context(
            patch("invarlock.cli.commands.run.detect_model_profile", detect_mlm)
        )
        for target in (
            "invarlock.cli.commands.run.resolve_tokenizer",
            "invarlock.cli.commands.run.resolve_tokenizer",
        ):
            stack.enter_context(
                patch(
                    target,
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
        run_command(
            config=str(cfg),
            device="cpu",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
        )


def test_baseline_mlm_no_masked_tokens_exit(tmp_path: Path):
    # Baseline provides labels but all -100 → no masked tokens; should exit in baseline pairing enforcement
    cfg = _base_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        """
{
  "meta": {"tokenizer_hash": "tokhash123"},
  "evaluation_windows": {
    "preview": {"window_ids": [0], "input_ids": [[1,2,3]], "attention_masks": [[1,1,1]], "labels": [[-100,-100,-100]]},
    "final": {"window_ids": [1], "input_ids": [[4,5,6]], "attention_masks": [[1,1,1]], "labels": [[-100,-100,-100]]}
  }
}
        """.strip()
    )

    class Cfg:
        def __init__(self):
            self.model = SimpleNamespace(adapter="hf_gpt2", id="gpt2", device="cpu")
            self.edit = SimpleNamespace(name="structured", plan={})
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
            self.guards = SimpleNamespace(order=[])
            self.eval = SimpleNamespace(
                spike_threshold=2.0, loss=SimpleNamespace(type="mlm", mask_prob=0.0)
            )
            self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
            self.output = SimpleNamespace(dir=tmp_path / "runs")

        def model_dump(self):
            return {}

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

    with ExitStack() as stack:
        stack.enter_context(patch("invarlock.cli.device.resolve_device", lambda d: d))
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
        stack.enter_context(patch("invarlock.cli.config.load_config", lambda p: Cfg()))
        stack.enter_context(
            patch("invarlock.cli.commands.run.detect_model_profile", detect_mlm)
        )
        for target in (
            "invarlock.cli.commands.run.resolve_tokenizer",
            "invarlock.cli.commands.run.resolve_tokenizer",
        ):
            stack.enter_context(
                patch(
                    target,
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
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                baseline=str(baseline),
                out=str(tmp_path / "runs"),
            )


@pytest.mark.parametrize(
    "order", list(permutations(["invariants", "spectral", "rmt"], 3))[:3]
)
def test_guard_order_permutations(tmp_path: Path, order):
    # Try a few guard order permutations to ensure no implicit ordering dependency
    cfg = _base_cfg(tmp_path)

    class Reg:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name, load_model=lambda model_id, device=None: object()
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, n, t):
            return {"name": n, "module": f"{t}.{n}", "version": "test"}

    class DummyCfg:
        def __init__(self):
            self.model = SimpleNamespace(id="gpt2", adapter="hf_gpt2", device="cpu")
            self.edit = SimpleNamespace(name="structured", plan={})
            self.auto = SimpleNamespace(enabled=False, tier="balanced", probes=0)
            self.guards = SimpleNamespace(order=list(order))
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

    with ExitStack() as stack:
        stack.enter_context(
            patch("invarlock.cli.config.load_config", lambda p: DummyCfg())
        )
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: Reg())
        )
        stack.enter_context(patch("invarlock.cli.device.resolve_device", lambda d: d))
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
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
        # Should not crash regardless of guard order
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
