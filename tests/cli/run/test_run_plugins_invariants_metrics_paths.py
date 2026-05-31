from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from invarlock.cli.commands.run import run_command


def _write_cfg(tmp_path: Path, preview=2, final=2, loss_type="auto") -> Path:
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
    type: {loss_type}

output:
  dir: runs
        """
    )
    return p


def _common_ce():
    return (
        patch(
            "invarlock.reporting.report_files.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
            },
        ),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
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
            "invarlock.cli.run_runtime_exec.detect_model_profile",
            lambda model_id=None, adapter=None: SimpleNamespace(
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
            "invarlock.cli.run_runtime_exec.resolve_tokenizer",
            lambda profile: (
                SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=50000),
                "tokhash123",
            ),
        ),
    )


def _provider_min():
    return SimpleNamespace(
        windows=lambda **kw: (
            SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
            SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
        )
    )


def _baseline_with_meta(tmp_path: Path, meta: dict, preview_ids, final_ids) -> Path:
    p = tmp_path / "baseline.json"
    payload = {
        "meta": meta,
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 1.0,
                "final": 1.0,
            }
        },
        "edit": {
            "name": "structured",
            "plan_digest": "baseline",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {
            "preview": {"window_ids": [0], "input_ids": preview_ids},
            "final": {"window_ids": [1], "input_ids": final_ids},
        },
    }
    p.write_text(json.dumps(payload))
    return p


def test_provider_non_evalwindow_mismatch_counts_no_exit(tmp_path: Path):
    cfg = _write_cfg(tmp_path, preview=2, final=1)

    class Provider:
        def windows(self, **kwargs):
            return SimpleNamespace(
                input_ids=[[1, 2]], attention_masks=[[1, 1]]
            ), SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]])

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
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )


def test_provider_indices_not_iterable_fallback(tmp_path: Path):
    cfg = _write_cfg(tmp_path)

    class Provider:
        def windows(self, **kwargs):
            prev = SimpleNamespace(
                input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]], indices=object()
            )
            fin = SimpleNamespace(
                input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]], indices=object()
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
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )


def test_metrics_merges_masked_totals_from_context(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    captured = {}

    class Runner:
        def execute(self, **kwargs):
            ctx = {
                "dataset_meta": {
                    "masked_tokens_total": 3,
                    "masked_tokens_preview": 1,
                    "masked_tokens_final": 2,
                }
            }
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context=ctx,
                status="success",
            )

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.reporting.report_files.save_report", cap_save)
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
    m = captured["r"]["metrics"]
    assert (
        m.get("masked_tokens_total") == 3
        and m.get("masked_tokens_preview") == 1
        and m.get("masked_tokens_final") == 2
    )


def test_metrics_optional_logloss_keys_persisted(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    captured = {}

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={
                    "ppl_preview": 1.0,
                    "ppl_final": 1.0,
                    "ppl_ratio": 1.0,
                    "logloss_preview": 2.0,
                    "logloss_final": 2.0,
                    "logloss_delta": 0.0,
                },
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    def cap_save(r, d, formats=None, filename_prefix=None):
        captured["r"] = r
        return {"json": str(d / (str(filename_prefix or "report") + ".json"))}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(
            patch("invarlock.reporting.report_files.save_report", cap_save)
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )
    m = captured["r"].get("metrics", {})
    assert "logloss_preview" in m and "logloss_final" in m and "logloss_delta" in m


def test_guard_overhead_fail_exits(tmp_path: Path):
    cfg = _write_cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 2.0, "ppl_ratio": 2.0},
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        # Patch validator to fail
        for target in (
            "invarlock.reporting.validate.validate_guard_overhead",
            "invarlock.cli.run_runtime_exec.validate_guard_overhead",
            "invarlock.cli.run_runtime_exec.validate_guard_overhead",
        ):
            stack.enter_context(
                patch(
                    target,
                    lambda *a, **k: SimpleNamespace(
                        passed=False,
                        messages=[],
                        warnings=[],
                        errors=[],
                        checks={},
                        metrics={"overhead_ratio": 2.0, "overhead_percent": 100.0},
                    ),
                )
            )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        with pytest.raises(click.exceptions.Exit):
            run_command(
                config=str(cfg),
                device="cpu",
                profile="ci",
                out=str(tmp_path / "runs"),
                until_pass=False,
            )


def test_drift_gate_fail_nonfatal(tmp_path: Path):
    cfg = _write_cfg(tmp_path)

    class Runner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 3.0, "ppl_ratio": 3.0},
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg), device="cpu", out=str(tmp_path / "runs"), until_pass=False
        )


def test_retry_controller_until_pass_two_attempts(tmp_path: Path):
    cfg = _write_cfg(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "meta": {"tokenizer_hash": "tokhash123"},
                "metrics": {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 1.0,
                        "final": 1.0,
                    }
                },
                "edit": {
                    "name": "structured",
                    "plan_digest": "baseline",
                    "deltas": {
                        "params_changed": 0,
                        "heads_pruned": 0,
                        "neurons_pruned": 0,
                        "layers_modified": 0,
                    },
                },
                "evaluation_windows": {
                    "preview": {"window_ids": [0], "input_ids": [[1, 2, 3]]},
                    "final": {"window_ids": [1], "input_ids": [[4, 5, 6]]},
                },
            }
        )
    )

    class Runner:
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
                    "paired_windows": 1,
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
                        "input_ids": [[4, 5, 6]],
                        "attention_masks": [[1, 1, 1]],
                    },
                },
                status="success",
            )

    class RC:
        def __init__(self, max_attempts=3, timeout=None, verbose=False):
            self.attempt_history = []

        def should_retry(self, passed):
            return not passed and len(self.attempt_history) < 1

        def record_attempt(self, attempt, result_summary, edit_config):
            self.attempt_history.append(result_summary)

        def get_attempt_summary(self):
            return {"total_attempts": len(self.attempt_history), "elapsed_time": 0.1}

    with ExitStack() as stack:
        for ctx in _common_ce():
            stack.enter_context(ctx)
        # first fail then pass
        results = [{"validation": {"gateA": False}}, {"validation": {"gateA": True}}]

        def make_cert(report, baseline_report):
            return results.pop(0)

        stack.enter_context(patch("invarlock.core.retry.RetryController", RC))
        stack.enter_context(
            patch("invarlock.cli.run_execution.build_evaluation_report", make_cert)
        )
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: _provider_min())
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            baseline=str(baseline),
            out=str(tmp_path / "runs"),
            until_pass=True,
            max_attempts=2,
        )
