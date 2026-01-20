from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import _plan_release_windows, run_command


def _base_cfg(tmp_path: Path, preview=2, final=2) -> Path:
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
            "invarlock.core.registry.get_registry",
            lambda: SimpleNamespace(
                get_adapter=lambda name: SimpleNamespace(
                    name=name, load_model=lambda model_id, device: object()
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


def test_plan_release_windows_requested_zero_target():
    plan = _plan_release_windows(
        {
            "available_unique": 800,
            "available_nonoverlap": 800,
            "total_tokens": 1_000_000,
            "dedupe_rate": 0.1,
        },
        requested_preview=0,
        requested_final=300,
        max_calibration=50,
        console=None,
    )
    assert plan["target_per_arm"] == 300
    assert plan["coverage_ok"] is True


def test_evaluation_window_provider_success(tmp_path: Path):
    from invarlock.eval.data import EvaluationWindow

    cfg = _base_cfg(tmp_path, 1, 1)

    class Provider:
        def windows(self, **kwargs):
            prev = EvaluationWindow(
                input_ids=[[1, 2]], attention_masks=[[1, 1]], indices=[0]
            )
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
                        context=getattr(k.get("config"), "context", {}),
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
            until_pass=False,
        )


def test_report_flags_guard_recovered(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={"spectral": {"passed": False, "metrics": {}}},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
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
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    assert captured["r"]["flags"].get("guard_recovered") is True
    assert captured["r"]["guards"][0]["name"] == "spectral"


def test_metrics_optional_keys_propagated(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={
                    "ppl_preview": 1.0,
                    "ppl_final": 1.0,
                    "ppl_ratio": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_match_fraction": 1.0,
                },
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )

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
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    m = captured["r"]["metrics"]
    assert m.get("window_overlap_fraction") == 0.0
    assert m.get("window_match_fraction") == 1.0


def test_guard_overhead_payload_present_ci(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    class Runner:
        def execute(self, **kwargs):
            # Return minimal valid reports for both bare and guarded runs
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={},
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        # Patch the validator at both locations to guarantee the module ref is hit
        for target in (
            "invarlock.reporting.validate.validate_guard_overhead",
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
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    gh = captured["r"].get("guard_overhead", {})
    assert isinstance(gh, dict) and gh.get("evaluated") is True


def test_tokenizer_digest_non_string_keys_in_vocab(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    class Tok:
        def get_vocab(self):
            return {1: "a", 2: "b"}

    def resolver(_):
        return Tok(), None

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.commands.run.resolve_tokenizer", resolver)
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]
                        ),
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
                        context=getattr(k.get("config"), "context", {}),
                        status="success",
                    )
                ),
            )
        )
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    # Accept hash recorded either in meta or data depending on implementation
    meta_hash = captured["r"].get("meta", {}).get("tokenizer_hash")
    data_hash = captured["r"].get("data", {}).get("tokenizer_hash")
    assert isinstance(meta_hash or data_hash, str)


def test_tokenizer_digest_exception_unknown(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    class Tok:
        def get_vocab(self):
            raise RuntimeError("boom")

    def resolver(_):
        # tokenizer hash None forces digest path
        return Tok(), None

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.commands.run.resolve_tokenizer", resolver)
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]
                        ),
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
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )

    # token digest fallback should land in data.tokenizer_hash
    assert captured["r"]["data"].get("tokenizer_hash") in {"unknown", None}


def test_tokenizer_digest_vocab_attribute_non_mapping(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)

    class Tok:
        # No get_vocab; has a 'vocab' attribute that is not a mapping
        def __init__(self):
            self.vocab = 5

    def resolver(_):
        return Tok(), None

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.cli.commands.run.resolve_tokenizer", resolver)
        )
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]
                        ),
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
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )


def test_baseline_not_found_fallback_to_dataset(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    missing = tmp_path / "no_such_baseline.json"
    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]
                        ),
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
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            baseline=str(missing),
            out=str(tmp_path / "runs"),
            until_pass=False,
        )


def test_baseline_adjust_counts_success(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 2, 2)
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

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={
                    "ppl_preview": 1.0,
                    "ppl_final": 1.0,
                    "ppl_ratio": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_match_fraction": 1.0,
                    "window_pairing_reason": None,
                    "paired_windows": 1,
                    "loss_type": "ce",
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
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            out=str(tmp_path / "runs"),
            baseline=str(baseline),
            until_pass=False,
        )


def test_baseline_tokenizer_hash_mismatch_exit(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash-OLD"},
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [1], "input_ids": [[4, 5, 6]]},
                },
            }
        )
    )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]
                        ),
                        SimpleNamespace(
                            input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]
                        ),
                    )
                ),
            )
        )
        # Force tokenizer hash to be 'tokhash123' so a mismatch with baseline occurs
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.resolve_tokenizer",
                lambda profile: (
                    SimpleNamespace(
                        eos_token="</s>", pad_token="</s>", vocab_size=50000
                    ),
                    "tokhash123",
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
                out=str(tmp_path / "runs"),
                baseline=str(baseline),
                until_pass=False,
            )


def test_preview_final_tokens_computed_when_missing_in_baseline_meta(tmp_path: Path):
    cfg = _base_cfg(tmp_path, 1, 1)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [1], "input_ids": [[4]]},
                },
            }
        )
    )

    captured = {}

    def cap_save(r, run_dir, formats, filename_prefix=None):
        captured["r"] = r
        return {"json": str(run_dir / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch(
                "invarlock.eval.data.get_provider",
                lambda *a, **k: SimpleNamespace(
                    windows=lambda **kw: (
                        SimpleNamespace(
                            input_ids=[[9, 9, 9]], attention_masks=[[1, 1, 1]]
                        ),
                        SimpleNamespace(input_ids=[[8]], attention_masks=[[1]]),
                    )
                ),
            )
        )

        def runner_factory():
            class R:
                def execute(self, **kwargs):
                    cfg_ctx = getattr(kwargs.get("config"), "context", {})
                    return SimpleNamespace(
                        edit={},
                        metrics={
                            "ppl_preview": 1.0,
                            "ppl_final": 1.0,
                            "ppl_ratio": 1.0,
                            "window_overlap_fraction": 0.0,
                            "window_match_fraction": 1.0,
                            "window_pairing_reason": None,
                            "paired_windows": 1,
                            "loss_type": "ce",
                        },
                        guards={},
                        context=cfg_ctx,
                        evaluation_windows={
                            "preview": {
                                "window_ids": [0],
                                "input_ids": [[1, 2, 3]],
                                "attention_masks": [[1, 1, 1]],
                            },
                            "final": {
                                "window_ids": [1],
                                "input_ids": [[4]],
                                "attention_masks": [[1]],
                            },
                        },
                        status="success",
                    )

            return R()

        stack.enter_context(patch("invarlock.core.runner.CoreRunner", runner_factory))
        stack.enter_context(patch("invarlock.reporting.report.save_report", cap_save))
        run_command(
            config=str(cfg),
            device="cpu",
            profile="release",
            out=str(tmp_path / "runs"),
            baseline=str(baseline),
            until_pass=False,
        )

    data = captured["r"]["data"]
    assert data.get("preview_total_tokens") == 3
    assert data.get("final_total_tokens") == 1


def test_provider_indices_fallback_iteration(tmp_path: Path):
    # Provider returns indices iterable that is convertible to list; ensure run doesn’t crash
    cfg = _base_cfg(tmp_path, 1, 1)

    class Provider:
        def windows(self, **kwargs):
            prev = SimpleNamespace(
                input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]], indices=(0,)
            )
            fin = SimpleNamespace(
                input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]], indices=(1,)
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
        run_command(
            config=str(cfg),
            device="cpu",
            profile=None,
            out=str(tmp_path / "runs"),
            until_pass=False,
        )
