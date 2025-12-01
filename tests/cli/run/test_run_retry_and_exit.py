# ruff: noqa: I001,E402,F811
from __future__ import annotations

# Consolidated retry, until-pass and exit-code behaviors

import json
import textwrap
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app as cli
from invarlock.cli.commands.run import run_command


# --------------------
# Content merged from test_run_command_retry.py
# --------------------


class DummyConfig:
    def __init__(self, output_dir: Path):
        self.model = SimpleNamespace(
            id="dummy-model", adapter="dummy-adapter", device="cpu"
        )
        self.edit = SimpleNamespace(name="dummy-edit", plan={"energy_keep": 0.98})
        self.auto = SimpleNamespace(
            enabled=True, tier="balanced", probes=0, target_pm_ratio=1.2
        )
        self.guards = SimpleNamespace(order=[])
        self.dataset = SimpleNamespace(
            provider=None,
            seq_len=16,
            stride=8,
            preview_n=1,
            final_n=1,
            split="validation",
            seed=42,
        )
        self.eval = SimpleNamespace(spike_threshold=2.0)
        self.output = SimpleNamespace(dir=output_dir)

    def model_dump(self) -> dict:
        return {
            "model": {
                "id": self.model.id,
                "adapter": self.model.adapter,
                "device": self.model.device,
            },
            "edit": {"name": self.edit.name, "plan": self.edit.plan},
            "auto": {
                "enabled": self.auto.enabled,
                "tier": self.auto.tier,
                "probes": self.auto.probes,
                "target_pm_ratio": self.auto.target_pm_ratio,
            },
            "guards": {"order": self.guards.order},
            "dataset": {
                "provider": self.dataset.provider,
                "seq_len": self.dataset.seq_len,
                "stride": self.dataset.stride,
                "preview_n": self.dataset.preview_n,
                "final_n": self.dataset.final_n,
                "seed": self.dataset.seed,
                "split": self.dataset.split,
            },
            "eval": {"spike_threshold": self.eval.spike_threshold},
            "output": {"dir": str(self.output.dir)},
        }


@pytest.fixture
def stubbed_run_environment(monkeypatch, tmp_path):
    attempts = {"guarded_runs": 0}
    adapter_calls = []

    class DummyAdapter:
        name = "dummy-adapter"

        def load_model(self, model_id: str, device: str):
            adapter_calls.append((model_id, device))
            return object()

        def snapshot(self, model):  # noqa: ARG002 - stub
            return {"ok": True}

        def restore(self, model, blob):  # noqa: ARG002 - stub
            return None

    class DummyEdit:
        name = "dummy-edit"

    class DummyRegistry:
        def __init__(self):
            self.adapter = DummyAdapter()
            self.edit = DummyEdit()

        def get_adapter(self, _name):
            return self.adapter

        def get_edit(self, _name):
            return self.edit

        def get_guard(self, _name):
            raise KeyError("no guards in test")

        def get_plugin_metadata(self, name, plugin_type):
            return {
                "name": name,
                "module": f"dummy.{plugin_type}.{name}",
                "version": "test",
                "available": True,
                "entry_point": None,
                "entry_point_group": None,
            }

    class DummyRunner:
        def __init__(self):
            self.executions = 0

        def execute(self, **kwargs):
            self.executions += 1
            ctx = getattr(kwargs.get("config"), "context", {}) or {}
            materialize = bool(ctx.get("materialize"))
            if materialize:
                attempts["guarded_runs"] += 1
            return SimpleNamespace(
                edit={"deltas": {}},
                metrics={
                    "ppl_preview": 1.0,
                    "ppl_final": 1.0,
                    "ppl_ratio": 1.0,
                },
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    # Registry and runner
    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr("invarlock.core.runner.CoreRunner", lambda: DummyRunner())
    # Provider windows
    monkeypatch.setattr(
        "invarlock.eval.data.get_provider",
        lambda *a, **k: SimpleNamespace(
            windows=lambda **kw: (
                SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]]),
                SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]]),
            )
        ),
    )
    # Device helpers
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda d: "cpu")
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
    )
    # Tokenizer
    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10),
            "tokhash123",
        ),
    )

    return attempts, adapter_calls


def test_run_command_retries_and_materializes_once(
    tmp_path: Path, stubbed_run_environment
):
    attempts, _adapter_calls = stubbed_run_environment
    cfg = DummyConfig(tmp_path)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg.model_dump()))
    # Simulate PASS then materialize branch with until_pass
    run_command(
        config=str(cfg_path),
        device="cpu",
        profile="ci",
        until_pass=True,
        max_attempts=2,
        out=str(tmp_path / "runs"),
    )
    assert attempts["guarded_runs"] <= 2


# --------------------
# Content merged from test_run_until_pass_retry.py
# --------------------


def test_run_command_until_pass_retry_flow(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
model:
  adapter: hf_gpt2
  id: gpt2
  device: cpu
edit:
  name: quant_rtn
  plan: {}

dataset:
  provider: synthetic
  id: synthetic
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

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"meta": {}, "metrics": {"ppl_preview": 10.0, "ppl_final": 10.0}})
    )

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name, load_model=lambda model_id, device: object()
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    def _runner_exec(**kwargs):
        return SimpleNamespace(
            edit={"deltas": {"params_changed": 0}},
            metrics={"loss_type": "ce"},
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={
                "preview": {"logloss": [3.0, 3.2], "token_counts": [10, 10]},
                "final": {"logloss": [3.1, 3.3], "token_counts": [10, 10]},
            },
            status="success",
        )

    with ExitStack() as stack:
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: DummyRegistry())
        )
        stack.enter_context(
            patch(
                "invarlock.core.runner.CoreRunner",
                lambda: SimpleNamespace(execute=_runner_exec),
            )
        )
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
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.detect_model_profile",
                lambda *a, **k: SimpleNamespace(
                    default_loss="ce",
                    invariants=[],
                    cert_lints=[],
                    module_selectors={},
                    family="gpt2",
                ),
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run.resolve_tokenizer",
                lambda *a, **k: (
                    SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10),
                    "tokhash123",
                ),
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            baseline=str(baseline),
            until_pass=True,
            max_attempts=2,
            out=str(tmp_path / "runs"),
        )


# --------------------
# Content merged from test_run_retry_summary.py
# --------------------


def test_until_pass_retry_summary_printed(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            model:
              adapter: hf_gpt2
              id: gpt2
              device: cpu
            edit:
              name: quant_rtn
              plan: {}

            dataset:
              provider: synthetic
              id: synthetic
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
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"meta": {"tokenizer_hash": "tokhash123"}}))

    summary_called = {"ok": False}

    class RC:
        def __init__(self, max_attempts=3, timeout=None, verbose=False):  # noqa: ARG002
            self.attempt_history: list[dict] = []

        def should_retry(self, passed):  # noqa: ARG002
            return False

        def record_attempt(self, attempt, result_summary, edit_config):  # noqa: ARG002
            self.attempt_history.append(result_summary)

        def get_attempt_summary(self):
            summary_called["ok"] = True
            return {"total_attempts": len(self.attempt_history), "elapsed_time": 0.1}

    class DummyRegistry:
        def get_adapter(self, name):  # noqa: ARG002
            return SimpleNamespace(
                name=name, load_model=lambda model_id, device: object()
            )

        def get_edit(self, name):  # noqa: ARG002
            return SimpleNamespace(name=name)

        def get_guard(self, name):  # noqa: ARG002
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):  # noqa: ARG002
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    class DummyRunner:
        def execute(self, **kwargs):  # noqa: ARG002
            return SimpleNamespace(
                edit={},
                metrics={"ppl_preview": 1.0, "ppl_final": 1.0, "ppl_ratio": 1.0},
                guards={},
                context={"dataset_meta": {}},
                status="success",
            )

    with ExitStack() as stack:
        stack.enter_context(
            patch("invarlock.core.registry.get_registry", lambda: DummyRegistry())
        )
        stack.enter_context(patch("invarlock.core.retry.RetryController", RC))
        stack.enter_context(
            patch("invarlock.core.runner.CoreRunner", lambda: DummyRunner())
        )
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
        stack.enter_context(patch("invarlock.cli.device.resolve_device", lambda d: d))
        stack.enter_context(
            patch(
                "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
            )
        )
        stack.enter_context(
            patch(
                "invarlock.reporting.report.save_report",
                lambda report, run_dir, formats, filename_prefix: {
                    "json": str(run_dir / (str(filename_prefix or "report") + ".json"))
                },
            )
        )
        stack.enter_context(
            patch(
                "invarlock.cli.commands.run._print_retry_summary",
                lambda console, rc: None,
            )
        )
        run_command(
            config=str(cfg),
            device="cpu",
            profile="ci",
            baseline=str(baseline),
            until_pass=True,
            max_attempts=1,
            out=str(tmp_path / "runs"),
        )


# --------------------
# Content merged from test_run_exit_codes.py
# --------------------


def _cfg(tmp_path: Path) -> str:
    yml = tmp_path / "cfg.yaml"
    yml.write_text(
        textwrap.dedent(
            """
            model:
              adapter: hf_gpt2
              id: gpt2
              device: auto
            edit:
              name: quant_rtn
              plan: {}

            dataset:
              provider: synthetic
              id: synthetic
              split: validation
              seq_len: 8
              stride: 4
              preview_n: 2
              final_n: 2

            guards:
              order: []

            eval:
              metric: { kind: ppl_causal, reps: 10, ci_level: 0.95 }
              loss: { type: auto }

            output:
              dir: runs
            """
        )
    )
    return str(yml)


def _stub_minimal_environment(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    monkeypatch.setattr("invarlock.cli.device.resolve_device", lambda d: "cpu")
    monkeypatch.setattr(
        "invarlock.cli.device.validate_device_for_config", lambda d: (True, "")
    )

    class DummyRegistry:
        def get_adapter(self, name):
            return SimpleNamespace(
                name=name,
                load_model=lambda model_id, device=None: object(),
                snapshot=lambda _m=None: b"blob",
                restore=lambda _m, _b=None: None,
            )

        def get_edit(self, name):
            return SimpleNamespace(name=name)

        def get_guard(self, name):
            raise KeyError("no guards")

        def get_plugin_metadata(self, name, plugin_type):
            return {"name": name, "module": f"{plugin_type}.{name}", "version": "test"}

    def _runner_exec(**kwargs):
        return SimpleNamespace(
            edit={"deltas": {"params_changed": 0}},
            metrics={
                "window_overlap_fraction": 0.0,
                "window_match_fraction": 1.0,
                "loss_type": "ce",
            },
            guards={},
            context={"dataset_meta": {}},
            evaluation_windows={
                "preview": {"logloss": [3.0, 3.2], "token_counts": [10, 10]},
                "final": {"logloss": [3.1, 3.3], "token_counts": [10, 10]},
            },
            status="success",
        )

    monkeypatch.setattr("invarlock.core.registry.get_registry", lambda: DummyRegistry())
    monkeypatch.setattr(
        "invarlock.core.runner.CoreRunner",
        lambda: SimpleNamespace(execute=_runner_exec),
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
        "invarlock.cli.commands.run.detect_model_profile",
        lambda model_id=None, adapter=None: SimpleNamespace(
            default_loss="ce",
            invariants=[],
            cert_lints=[],
            module_selectors={},
            family="test",
        ),
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.run.resolve_tokenizer",
        lambda *a, **k: (
            SimpleNamespace(eos_token="</s>", pad_token="</s>", vocab_size=10),
            "tokhash123",
        ),
    )


def test_schema_invalid_returns_2(tmp_path: Path, monkeypatch):
    _stub_minimal_environment(monkeypatch, tmp_path)
    cfg = Path(_cfg(tmp_path))
    monkeypatch.setattr(
        "invarlock.eval.primary_metric.compute_primary_metric_from_report",
        lambda *a, **k: {},
    )
    r = CliRunner().invoke(cli, ["run", "-c", str(cfg), "--profile", "dev"])
    assert r.exit_code == 2
    assert "schema" in (r.stdout).lower()


def test_parity_error_dev_exit_1(tmp_path: Path, monkeypatch):
    _stub_minimal_environment(monkeypatch, tmp_path)
    cfg = Path(_cfg(tmp_path))
    from invarlock.cli.commands import run as runmod

    monkeypatch.setattr(
        runmod, "InvarlockError", type("InvarlockError", (Exception,), {})
    )
    monkeypatch.setattr(
        runmod,
        "detect_model_profile",
        lambda *a, **k: (_ for _ in ()).throw(runmod.InvarlockError()),
    )
    r = CliRunner().invoke(cli, ["run", "-c", str(cfg), "--profile", "dev"])
    assert r.exit_code == 1


def test_parity_error_ci_exit_3(tmp_path: Path, monkeypatch):
    _stub_minimal_environment(monkeypatch, tmp_path)
    cfg = Path(_cfg(tmp_path))
    from invarlock.cli.commands import run as runmod

    monkeypatch.setattr(
        runmod, "InvarlockError", type("InvarlockError", (Exception,), {})
    )
    monkeypatch.setattr(
        runmod,
        "detect_model_profile",
        lambda *a, **k: (_ for _ in ()).throw(runmod.InvarlockError()),
    )
    r = CliRunner().invoke(cli, ["run", "-c", str(cfg), "--profile", "ci"])
    assert r.exit_code == 3
