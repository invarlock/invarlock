from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command

_SNS = SimpleNamespace


def _write_base_cfg(tmp_path: Path, preview_n=2, final_n=2) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
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
  preview_n: {preview_n}
  final_n: {final_n}

guards:
  order: []

eval:
  loss:
    type: auto

output:
  dir: runs
        """
    )
    return cfg


def _common_patches_ce():
    return (
        patch(
            "invarlock.core.registry.get_registry",
            lambda: SimpleNamespace(
                get_adapter=lambda name: SimpleNamespace(
                    name=name, load_model=lambda model_id, device: object()
                ),
                get_edit=lambda name: SimpleNamespace(name=name),
                get_guard=lambda name: (_ for _ in ()).throw(KeyError("no guard")),
                get_plugin_metadata=lambda n, t: {
                    "name": n,
                    "module": f"{t}.{n}",
                    "version": "test",
                },
            ),
        ),
        patch(
            "invarlock.cli.run_runtime.detect_model_profile",
            lambda model_id, adapter: SimpleNamespace(
                default_loss="ce",
                default_provider=None,
                default_metric=None,
                model_id=model_id,
                adapter=adapter,
                family="gpt2",
                module_selectors={},
                invariants=[],
                cert_lints=[],
            ),
        ),
        patch(
            "invarlock.cli.run_runtime.resolve_tokenizer",
            lambda model_profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report_files.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (filename_prefix + ".json"))
            },
        ),
    )


def _common_patches_mlm():
    return (
        patch(
            "invarlock.core.registry.get_registry",
            lambda: SimpleNamespace(
                get_adapter=lambda name: SimpleNamespace(
                    name=name, load_model=lambda model_id, device: object()
                ),
                get_edit=lambda name: SimpleNamespace(name=name),
                get_guard=lambda name: (_ for _ in ()).throw(KeyError("no guard")),
                get_plugin_metadata=lambda n, t: {
                    "name": n,
                    "module": f"{t}.{n}",
                    "version": "test",
                },
            ),
        ),
        patch(
            "invarlock.cli.run_runtime.detect_model_profile",
            lambda model_id, adapter: SimpleNamespace(
                default_loss="mlm",
                default_provider=None,
                default_metric=None,
                model_id=model_id,
                adapter=adapter,
                family="bert",
                module_selectors={},
                invariants=[],
                cert_lints=[],
            ),
        ),
        patch(
            "invarlock.cli.run_runtime.resolve_tokenizer",
            lambda model_profile: (
                SimpleNamespace(
                    mask_token_id=103,
                    eos_token="</s>",
                    pad_token="</s>",
                    vocab_size=50000,
                    all_special_ids=[0, 1, 2],
                ),
                "tokhash123",
            ),
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report_files.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (filename_prefix + ".json"))
            },
        ),
    )


def _compute_seq_hash(seqs: list[list[int]]) -> str:
    import hashlib
    from array import array

    h = hashlib.blake2s(digest_size=16)
    for seq in seqs:
        h.update(len(seq).to_bytes(4, "little", signed=False))
        arr = array("I", (int(tok) & 0xFFFFFFFF for tok in seq))
        h.update(arr.tobytes())
    return h.hexdigest()


# --------------------
# Merged from test_run_branch_supplemental.py
# --------------------


def _supp_cfg(tmp_path: Path, preview=1, final=1) -> Path:
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


def _supp_common_patches_detect_ce():
    return (
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.reporting.report_files.save_report",
            lambda report, run_dir, formats, filename_prefix=None: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
        patch(
            "invarlock.cli.run_runtime.detect_model_profile",
            lambda model_id=None, adapter=None: _SNS(
                default_loss="ce",
                model_id=model_id,
                adapter=adapter,
                module_selectors={},
                invariants=set(),
                cert_lints=[],
                family="gpt2",
            ),
        ),
    )


def _supp_provider_min():
    return _SNS(
        windows=lambda **kw: (
            _SNS(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            _SNS(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )
    )


def test_module_selectors_not_overridden_when_present(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  adapter: hf_causal
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan:
    module_selectors: {heads: [0]}

dataset:
  provider: synthetic
  split: validation
  seq_len: 8
  stride: 4
  preview_n: 1
  final_n: 1

guards:
  order: []

eval:
  loss:
    type: auto

output:
  dir: runs
        """
    )

    captured = {}

    def runner_factory():
        class R:
            def execute(self, **kwargs):
                captured["edit_config"] = kwargs.get("edit_config")
                return SimpleNamespace(
                    edit={},
                    metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                    guards={},
                    context={"dataset_meta": {}},
                )

        return R()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime.detect_model_profile",
                lambda model_id, adapter: SimpleNamespace(
                    default_loss="ce",
                    model_id=model_id,
                    adapter=adapter,
                    module_selectors={"attn": ["q_proj"]},
                    invariants=set(),
                    cert_lints=[],
                ),
            )
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", runner_factory))
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
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    # Should preserve user-specified selectors
    assert captured.get("edit_config", {}).get("module_selectors") == {"heads": [0]}


def test_skip_missing_guard_path(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("ok: true")

    captured = {}

    def runner_factory():
        class R:
            def execute(self, **kwargs):
                captured["guards"] = kwargs.get("guards")
                return SimpleNamespace(
                    edit={},
                    metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                    guards={},
                    context={"dataset_meta": {}},
                )

        return R()

    with ExitStack() as stack:
        # Bypass config validation to allow unknown guard
        class DummyCfg:
            def __init__(self, outdir):
                self.model = SimpleNamespace(
                    id="gpt2", adapter="hf_causal", device="cpu"
                )
                self.edit = SimpleNamespace(name="quant_rtn", plan={})
                self.auto = SimpleNamespace(
                    enabled=False, tier="balanced", probes=0, target_pm_ratio=None
                )
                self.guards = SimpleNamespace(order=["missing_guard"])  # non-existent
                self.dataset = SimpleNamespace(
                    provider="synthetic",
                    seq_len=8,
                    stride=4,
                    preview_n=1,
                    final_n=1,
                    split="validation",
                )
                self.eval = SimpleNamespace(
                    spike_threshold=2.0, loss=SimpleNamespace(type="auto")
                )
                self.output = SimpleNamespace(dir=outdir)

            def model_dump(self):
                return {}

        stack.enter_context(
            patch(
                "invarlock.core.config_loader.load_config",
                lambda p: DummyCfg(tmp_path / "runs"),
            )
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", runner_factory))
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
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert captured.get("guards") == []


def test_dedupe_duplicate_windows_raises_exit(tmp_path: Path):
    # Force duplicate windows so dedupe branch triggers and raises error (non-release profile)
    cfg = _write_base_cfg(tmp_path, preview_n=4, final_n=4)

    class Provider:
        def windows(self, **kwargs):
            # Many identical windows -> duplicates
            prev = SimpleNamespace(
                input_ids=[[1, 2, 3]] * 4, attention_masks=[[1, 1, 1]] * 4
            )
            fin = SimpleNamespace(
                input_ids=[[1, 2, 3]] * 4, attention_masks=[[1, 1, 1]] * 4
            )
            return prev, fin

    with ExitStack() as stack:
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
                    )
                ),
            )
        )
        for ctx in _common_patches_ce():
            stack.enter_context(ctx)
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile=None,
                out=str(tmp_path / "runs"),
                until_pass=False,
            )
