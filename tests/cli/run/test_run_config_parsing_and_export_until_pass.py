from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.cli.run.test_run_config_parsing_and_export import (
    _Cfg,
    _core_report,
    _detect_profile,
    _tok,
    run_command,
)


def test_run_command_until_pass_auto_tune_head_budget_paths(tmp_path: Path) -> None:
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
                        "input_ids": [[1, 2]],
                        "attention_masks": [[1, 1]],
                    },
                    "final": {
                        "window_ids": [1],
                        "input_ids": [[3, 4]],
                        "attention_masks": [[1, 1]],
                    },
                },
            }
        )
    )

    class Adapter:
        name = "hf_causal"

        def load_model(self, model_id: str, device: str | None = None):  # noqa: ARG002
            return object()

    adapter = Adapter()

    class Registry:
        def get_adapter(self, name):  # noqa: ARG002
            return adapter

        def get_edit(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_guard(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_plugin_metadata(self, name, plugin_type):  # noqa: ARG002
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    class RC:
        def __init__(self, *args, **kwargs):  # noqa: ANN001,ARG002
            self.attempt_history = []

        def record_attempt(self, attempt, result, edit_config):  # noqa: ANN001
            self.attempt_history.append((attempt, result, edit_config))

        def should_retry(self, passed: bool) -> bool:  # noqa: ARG002
            return False

    def exec_stub(**kwargs):  # noqa: ANN001
        return _core_report(
            evaluation_windows={
                "preview": {"input_ids": [[1, 2]]},
                "final": {"input_ids": [[3, 4]]},
            }
        ), kwargs.get("model")

    def post_stub(**kwargs):  # noqa: ANN001
        return {"json": str(tmp_path / "report.json")}

    cfg = _Cfg(
        outdir=tmp_path / "runs",
        dataset_provider="synthetic",
        edit_plan={
            "heads": {
                "mask_only": True,
                "_auto_search": {"keep_low": 0, "keep_high": 8, "keep_current": 4},
            }
        },
    )

    def cert_fail_pm(_report, _baseline_report):  # noqa: ANN001
        return {"validation": {"primary_metric_acceptable": False, "drift_ok": True}}

    def cert_fail_other(_report, _baseline_report):  # noqa: ANN001
        return {"validation": {"primary_metric_acceptable": True, "drift_ok": False}}

    for make_cert in (cert_fail_pm, cert_fail_other):
        with ExitStack() as stack:
            for p in (
                patch(
                    "invarlock.cli.run_config.prepare_config_for_run",
                    lambda **k: cfg,
                ),
                patch(
                    "invarlock.cli.run_runtime.detect_model_profile", _detect_profile
                ),
                patch(
                    "invarlock.cli.run_runtime.resolve_tokenizer",
                    lambda *_a, **_k: _tok(),
                ),
                patch("invarlock.cli.device.resolve_device", lambda d: d),
                patch(
                    "invarlock.cli.device.validate_device_for_config",
                    lambda d: (True, ""),
                ),
                patch("invarlock.core.registry.get_registry", lambda: Registry()),
                patch(
                    "invarlock.core.run_orchestrator_execute._should_measure_overhead_impl",
                    lambda *_a: (False, False, None),
                ),
                patch("invarlock.cli.run_runtime_exec.execute_guarded_run", exec_stub),
                patch(
                    "invarlock.cli.run_artifact_output.postprocess_and_summarize",
                    post_stub,
                ),
                patch("invarlock.core.retry.RetryController", RC),
                patch("invarlock.cli.run_execution.build_evaluation_report", make_cert),
            ):
                stack.enter_context(p)
            run_command(
                config="dummy.yaml",
                device="cpu",
                profile=None,
                out=str(tmp_path / "runs"),
                baseline=str(baseline),
                until_pass=True,
                max_attempts=1,
            )
