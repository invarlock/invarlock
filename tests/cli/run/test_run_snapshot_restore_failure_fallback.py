from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

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


def test_until_pass_restore_failure_discards_model_and_reloads_next_attempt(
    tmp_path: Path,
) -> None:
    cfg_path = _base_cfg(tmp_path)
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
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class Adapter:
        name = "hf_causal"

        def __init__(self) -> None:
            self.load_calls = 0
            self.restore_calls = 0

        def load_model(
            self, model_id: str, device: str | None = None
        ) -> SimpleNamespace:
            self.load_calls += 1
            return SimpleNamespace(
                named_parameters=lambda: [], named_buffers=lambda: []
            )

        def snapshot_chunked(self, model: object) -> str:
            return str(tmp_path / "snapdir")

        def restore_chunked(self, model: object, path: str) -> None:
            self.restore_calls += 1
            if self.restore_calls >= 2:
                raise RuntimeError("restore failed")

    adapter = Adapter()
    execute_models: list[object] = []

    class Runner:
        def execute(self, **kwargs: Any) -> SimpleNamespace:
            execute_models.append(kwargs.get("model"))
            return SimpleNamespace(
                edit={"plan_digest": "abcd", "deltas": {"params_changed": 0}},
                metrics={
                    "window_overlap_fraction": 0.0,
                    "window_match_fraction": 1.0,
                    "window_pairing_reason": None,
                    "paired_windows": 1,
                    "loss_type": "ce",
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 10.0,
                        "final": 10.0,
                    },
                },
                guards={},
                context={"dataset_meta": {"tokenizer_hash": "tokhash123"}},
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

    class RetryController:
        def __init__(
            self,
            max_attempts: int = 3,
            timeout: float | None = None,
            verbose: bool = False,
        ) -> None:
            self.attempt_history: list[dict[str, object]] = []

        def should_retry(self, passed: bool) -> bool:
            return (not passed) and len(self.attempt_history) < 3

        def record_attempt(
            self,
            attempt: int,
            result_summary: dict[str, object],
            edit_config: object,
        ) -> None:
            self.attempt_history.append(dict(result_summary))

        def get_attempt_summary(self) -> dict[str, float | int]:
            return {"total_attempts": len(self.attempt_history), "elapsed_time": 0.1}

    cert_calls = {"n": 0}

    def make_report(
        report: object, baseline_report: object
    ) -> dict[str, dict[str, bool]]:
        cert_calls["n"] += 1
        if cert_calls["n"] == 1:
            return {"validation": {"primary_metric_acceptable": False}}
        return {"validation": {"primary_metric_acceptable": True}}

    def save_report(
        report: object,
        run_dir: str | Path,
        formats: object,
        filename_prefix: str,
    ) -> dict[str, str]:
        captured["report"] = report
        return {"json": str(Path(run_dir) / f"{filename_prefix}.json")}

    class Provider:
        def windows(self, **kwargs: Any) -> tuple[SimpleNamespace, SimpleNamespace]:
            return (
                SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]]),
                SimpleNamespace(input_ids=[[4, 5, 6]], attention_masks=[[1, 1, 1]]),
            )

    with ExitStack() as stack:
        stack.enter_context(patch("invarlock.cli.device.resolve_device", lambda d: d))
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
        stack.enter_context(
            patch("invarlock.core.retry.RetryController", RetryController)
        )
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
        stack.enter_context(patch("invarlock.core.runner.CoreRunner", lambda: Runner()))
        stack.enter_context(
            patch("invarlock.eval.data.get_provider", lambda *a, **k: Provider())
        )
        stack.enter_context(
            patch(
                "invarlock.cli.run_execution.build_evaluation_report",
                make_report,
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.detect_model_profile",
                lambda *a, **k: SimpleNamespace(
                    default_loss="ce",
                    default_provider=None,
                    default_metric=None,
                    family="test",
                    module_selectors={},
                    invariants=[],
                    cert_lints=[],
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.run_runtime_exec.resolve_tokenizer",
                lambda *a, **k: (
                    SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10),
                    "tokhash123",
                ),
            )
        )
        stack.enter_context(
            patch("invarlock.reporting.report_bundle.save_report", save_report)
        )

        run_command(
            config=str(cfg_path),
            device="cpu",
            baseline=str(baseline),
            until_pass=True,
            max_attempts=3,
            out=str(tmp_path / "runs"),
        )

    assert adapter.restore_calls == 2
    assert adapter.load_calls == 2
    assert len(execute_models) == 2
    assert execute_models[0] is not execute_models[1]

    report = captured.get("report")
    assert isinstance(report, dict)
    prov = report.get("provenance")
    assert isinstance(prov, dict)
    assert prov.get("restore_failed") is True
    assert prov.get("reload_path_used") is True
