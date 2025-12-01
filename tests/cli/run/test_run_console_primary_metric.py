from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from invarlock import security
from invarlock.cli.commands.run import run_command


@pytest.mark.unit
def test_run_prints_primary_metric_not_legacy(tmp_path: Path, capsys):
    # Minimal YAML config
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

    class DummyRunner:
        def execute(self, **kwargs):
            # Minimal report with legacy fields (still present internally)
            return SimpleNamespace(
                edit={"deltas": {}, "plan_digest": "x"},
                metrics={
                    "ppl_preview": 10.0,
                    "ppl_final": 10.0,
                    "ppl_ratio": 1.0,
                },
                guards={},
                context={"dataset_meta": {}},
                evaluation_windows={
                    "preview": {"logloss": [1.0], "token_counts": [1]},
                    "final": {"logloss": [1.0], "token_counts": [1]},
                },
                status="success",
            )

    class StubProvider:
        def windows(self, **kwargs):
            prev = SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]])
            fin = SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]])
            return prev, fin

    # Force compute_primary_metric_from_report to a fixed payload to make
    # output deterministic
    def fake_pm(report, *, kind, baseline=None):
        return {"kind": kind, "preview": 12.3, "final": 12.1, "ratio_vs_baseline": 1.01}

    security.enforce_network_policy(True)
    with (
        patch(
            "invarlock.core.registry.get_registry",
            lambda: SimpleNamespace(
                get_adapter=lambda name: SimpleNamespace(
                    name=name, load_model=lambda model_id, device: object()
                ),
                get_edit=lambda name: SimpleNamespace(name=name),
                get_guard=lambda name: (_ for _ in ()).throw(KeyError("no guards")),
                get_plugin_metadata=lambda name, t: {
                    "name": name,
                    "module": f"{t}.{name}",
                    "version": "test",
                },
            ),
        ),
        patch("invarlock.core.runner.CoreRunner", lambda: DummyRunner()),
        patch("invarlock.eval.data.get_provider", lambda *a, **k: StubProvider()),
        patch("invarlock.cli.device.resolve_device", lambda d: d),
        patch("invarlock.cli.device.validate_device_for_config", lambda d: (True, "")),
        patch(
            "invarlock.eval.primary_metric.compute_primary_metric_from_report", fake_pm
        ),
        patch(
            "invarlock.reporting.report.save_report",
            lambda report, run_dir, formats, filename_prefix: {
                "json": str(run_dir / (filename_prefix + ".json"))
            },
        ),
    ):
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    out = capsys.readouterr().out
    assert "Primary Metric" in out
    assert "Drift (final/preview)" not in out
    assert "Final PPL" not in out
