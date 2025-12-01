from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from invarlock.cli.commands.run import run_command


@pytest.mark.unit
@pytest.mark.parametrize(
    "metric_kind", ["ppl_causal", "ppl_mlm", "ppl_seq2seq", "accuracy"]
)
def test_run_report_pm_only_no_ppl_keys(tmp_path: Path, metric_kind: str):
    cfg = tmp_path / "config.yaml"
    # minimal config; set eval.metric.kind to drive PM family deterministically
    cfg.write_text(
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
  preview_n: 1
  final_n: 1

guards:
  order: []

eval:
  loss:
    type: auto
  metric:
    kind: {metric_kind}

output:
  dir: runs
"""
    )

    # Dummy runner returns legacy ppl metrics internally; runner must not write them out
    class DummyRunner:
        def execute(self, **kwargs):
            return SimpleNamespace(
                edit={"deltas": {}, "plan_digest": "x"},
                metrics={
                    "ppl_preview": 10.0,
                    "ppl_final": 10.0,
                    "ppl_ratio": 1.0,
                    "ppl_preview_ci": (9.5, 10.5),
                    "ppl_final_ci": (9.5, 10.5),
                    "ppl_ratio_ci": (0.9, 1.1),
                },
                guards={},
                context={
                    "dataset_meta": {},
                    # Provide a minimal window_plan to surface stats in the run report
                    "window_plan": {
                        "requested_preview": 1,
                        "requested_final": 1,
                        "actual_preview": 1,
                        "actual_final": 1,
                        "coverage_ok": True,
                        "capacity": {"total_tokens": 32},
                    },
                },
                evaluation_windows={
                    "preview": {
                        "logloss": [1.0],
                        "token_counts": [1],
                        "window_ids": [1],
                        "input_ids": [[1, 2]],
                        "attention_masks": [[1, 1]],
                    },
                    "final": {
                        "logloss": [1.0],
                        "token_counts": [1],
                        "window_ids": [2],
                        "input_ids": [[3, 4]],
                        "attention_masks": [[1, 1]],
                    },
                },
                status="success",
            )

    class StubProvider:
        def windows(self, **kwargs):
            # Not used because DummyRunner produces evaluation_windows
            prev = SimpleNamespace(input_ids=[[1, 2]], attention_masks=[[1, 1]])
            fin = SimpleNamespace(input_ids=[[3, 4]], attention_masks=[[1, 1]])
            return prev, fin

    def fake_pm(report, *, kind, baseline=None):
        # Provide display_ci to mimic schema invariants
        return {
            "kind": kind,
            "preview": 12.3,
            "final": 12.1,
            "ratio_vs_baseline": 1.01 if kind != "accuracy" else 0.0,
            "display_ci": (12.0, 12.5),
        }

    captured = {}

    def fake_save_report(report, run_dir, formats, filename_prefix):
        captured["report"] = report
        return {"json": str(Path(run_dir) / (filename_prefix + ".json"))}

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
        patch("invarlock.reporting.report.save_report", fake_save_report),
    ):
        run_command(config=str(cfg), device="cpu", out=str(tmp_path / "runs"))

    report = captured["report"]
    assert isinstance(report, dict)
    # PM present
    pm = report.get("metrics", {}).get("primary_metric")
    assert isinstance(pm, dict) and pm.get("kind") == metric_kind
    assert "preview" in pm and "final" in pm and "ratio_vs_baseline" in pm
    # No ppl_* writes in run report metrics
    for key in (report.get("metrics", {}) or {}).keys():
        assert not key.startswith("ppl_"), f"unexpected legacy key {key}"
    # Provider digest and dataset split present
    prov = report.get("provenance", {})
    assert isinstance(prov.get("dataset_split"), str)
    assert isinstance(prov.get("provider_digest"), dict)
    # Pairing/window stats present under metrics.stats when window_plan exists
    stats = report.get("metrics", {}).get("stats", {})
    assert isinstance(stats, dict)
    assert "requested_preview" in stats and "requested_final" in stats
