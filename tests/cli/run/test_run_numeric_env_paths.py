from __future__ import annotations

from contextlib import ExitStack
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command
from tests.cli.run._support_run_common import (
    assert_single_run_output_artifacts,
    common_ce_detect_ce_patches,
    offline_registry_patch,
    write_base_run_config,
)
from tests.cli.run._support_run_common import (
    synthetic_provider_min as _provider_min,
)


def _base_cfg(tmp_path: Path, preview=1, final=1) -> Path:
    return write_base_run_config(
        tmp_path,
        preview,
        final,
        eval_fields="  spike_threshold: 2.0\n",
    )


@pytest.fixture(autouse=True)
def _offline_registry_stub():
    with offline_registry_patch():
        yield


def _common_ce_detect_ce():
    return common_ce_detect_ce_patches()


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
            self.model = SimpleNamespace(adapter="hf_causal", id="gpt2", device="cpu")
            self.edit = SimpleNamespace(name="quant_rtn", plan={})
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
        stack.enter_context(
            patch("invarlock.core.config_loader.load_config", lambda p: Cfg())
        )
        stack.enter_context(
            patch("invarlock.cli.run_runtime_exec.detect_model_profile", detect_mlm)
        )
        for target in (
            "invarlock.cli.run_runtime_exec.resolve_tokenizer",
            "invarlock.cli.run_runtime_exec.resolve_tokenizer",
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
            self.model = SimpleNamespace(id="gpt2", adapter="hf_causal", device="cpu")
            self.edit = SimpleNamespace(name="quant_rtn", plan={})
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
            patch("invarlock.core.config_loader.load_config", lambda p: DummyCfg())
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
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.resolve_tokenizer",
                lambda *_a, **_k: (
                    SimpleNamespace(
                        eos_token="</s>",
                        pad_token="</s>",
                        vocab_size=50000,
                    ),
                    "tokhash123",
                ),
            )
        )
        # Should not crash regardless of guard order
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))
    assert_single_run_output_artifacts(tmp_path)
