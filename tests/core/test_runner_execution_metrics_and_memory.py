from __future__ import annotations

from typing import Any

import pytest

from invarlock.core.api import Guard, ModelAdapter
from invarlock.core.runner import CoreRunner
from tests.core._support_runner_execution import (
    DummyAdapter,
    DummyEdit,
    DummyModel,
    GoodGuard,
    make_config,
)


def test_runner_rollback_on_guard_failure(monkeypatch, tmp_path):
    runner = CoreRunner()

    # Apply guard policy via resolver
    def fake_resolver(tier: str, edit_name: str | None, overrides: dict[str, Any]):
        assert tier in {"balanced", "aggressive", "conservative"}
        return {
            "good": {
                "alpha": 0.1
            },  # attribute set on guard.config via _apply_guard_policy
            "bad": {
                "beta": 0.2
            },  # attribute set on guard.policy via _apply_guard_policy
        }

    import invarlock.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "resolve_tier_policies", fake_resolver)

    class ValidateBoomGuard(Guard):
        name = "bad"

        def __init__(self):
            self.policy = {}

        def set_run_context(self, report):
            self.policy["context"] = True

        def validate(self, model: Any, adapter: ModelAdapter, context: dict[str, Any]):
            raise RuntimeError("validate boom")

    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit(name="edit-name")
    guards = [ValidateBoomGuard(), GoodGuard()]
    cfg = make_config(tmp_path)

    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)

    assert report.status == "failed"
    assert isinstance(report.error, str) and "validate boom" in report.error
    assert model._restored is True


def test_runner_success_and_catastrophic_spike(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)

    # Force eval metrics to a successful ratio
    def fake_eval_success(model, adapter, calib, report, preview_n, final_n, config):
        # CoreRunner._eval_phase normally returns only metrics dict; keep same shape here
        return {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}}

    # And a catastrophic spike case
    def fake_eval_spike(model, adapter, calib, report, preview_n, final_n, config):
        return {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 5.0}}

    monkeypatch.setattr(CoreRunner, "_eval_phase", staticmethod(fake_eval_success))
    report_ok = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    assert report_ok.status == "success"

    # Swap in the spike behavior
    monkeypatch.setattr(CoreRunner, "_eval_phase", staticmethod(fake_eval_spike))
    report_spike = runner.execute(
        model, adapter, edit, guards, cfg, calibration_data=None
    )
    assert report_spike.status == "rollback"
    assert "catastrophic" in report_spike.meta.get("rollback_reason", "")


def test_runner_edit_cannot_apply_sets_failed(tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit(can=False)
    guards = []
    cfg = make_config(tmp_path)

    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    assert report.status == "failed"
    assert isinstance(report.error, str) and "cannot be applied" in report.error


def test_runner_records_timing_and_guard_timings(tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path, checkpoint_interval=0)

    report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=None)
    timings = report.metrics.get("timings", {})
    assert isinstance(timings, dict)
    for key in (
        "prepare",
        "prepare_guards",
        "edit",
        "guards",
        "eval",
        "finalize",
        "total",
    ):
        assert key in timings

    guard_timings = report.metrics.get("guard_timings", {})
    assert isinstance(guard_timings, dict)
    assert guard_timings.get("good") is not None


def test_runner_merges_memory_snapshots_into_metrics(monkeypatch, tmp_path):
    import invarlock.core.runner_execution_plan as runner_exec_mod

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    cfg = make_config(tmp_path, checkpoint_interval=0)

    monkeypatch.setattr(
        runner_exec_mod,
        "capture_memory_snapshot",
        lambda phase: {"phase": phase, "rss_mb": 1.0},
    )
    monkeypatch.setattr(
        runner_exec_mod,
        "summarize_memory_snapshots",
        lambda _snaps: {"memory_mb_peak": 0.5},
    )

    report = runner.execute(model, adapter, edit, [], cfg, calibration_data=None)
    assert report.metrics.get("memory_snapshots")
    assert report.metrics.get("memory_mb_peak") is not None


def test_runner_skips_empty_memory_snapshots(monkeypatch, tmp_path):
    import invarlock.core.runner_execution_plan as runner_exec_mod

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    cfg = make_config(tmp_path, checkpoint_interval=0)

    monkeypatch.setattr(runner_exec_mod, "capture_memory_snapshot", lambda phase: {})
    monkeypatch.setattr(
        runner_exec_mod, "summarize_memory_snapshots", lambda _snaps: {}
    )

    report = runner.execute(model, adapter, edit, [], cfg, calibration_data=None)
    assert "memory_snapshots" not in report.metrics


def test_runner_memory_snapshot_summary_empty(monkeypatch, tmp_path):
    import invarlock.core.runner_execution_plan as runner_exec_mod

    called = {"summary": 0}

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    cfg = make_config(tmp_path, checkpoint_interval=0)

    monkeypatch.setattr(
        runner_exec_mod,
        "capture_memory_snapshot",
        lambda phase: {"phase": phase, "rss_mb": 1.0},
    )
    monkeypatch.setattr(
        runner_exec_mod,
        "summarize_memory_snapshots",
        lambda _snaps: called.__setitem__("summary", called["summary"] + 1) or {},
    )

    report = runner.execute(model, adapter, edit, [], cfg, calibration_data=None)
    assert report.metrics.get("memory_snapshots")
    assert called["summary"] == 1


def test_runner_eval_phase_with_calibration_uses_compute(monkeypatch, tmp_path):
    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)

    # Patch compute_real_metrics to avoid heavy torch usage and still exercise branch
    def fake_compute(model, calibration_data, adapter, preview_n, final_n, config):
        return (
            {
                "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.1},
                "latency_ms_per_tok": 0.0,
                "memory_mb_peak": 0.0,
            },
            {"preview": {"window_ids": [0]}, "final": {"window_ids": [0]}},
        )

    monkeypatch.setattr(CoreRunner, "_compute_real_metrics", staticmethod(fake_compute))

    # Minimal calibration data sample
    calibration = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1], "labels": [4, 5, 6]},
    ]

    report = runner.execute(
        model, adapter, edit, guards, cfg, calibration_data=calibration
    )
    assert report.status in {"success", "rollback"}  # depends on ppl thresholds
    assert "evaluation_windows" in report.__dict__
    assert isinstance(report.evaluation_windows.get("preview"), dict)


def test_eval_phase_debug_snapshot(monkeypatch, tmp_path):
    import os

    runner = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    edit = DummyEdit()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)
    # Provide small calibration data and patch compute to avoid heavy logic
    cal = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
    os.environ["INVARLOCK_DEBUG_TRACE"] = "1"
    try:
        monkeypatch.setattr(
            CoreRunner,
            "_compute_real_metrics",
            staticmethod(
                lambda *a, **k: (
                    {
                        "primary_metric": {
                            "kind": "ppl_causal",
                            "preview": 1.0,
                            "final": 1.0,
                        }
                    },
                    {"preview": {}, "final": {}},
                )
            ),
        )
        report = runner.execute(model, adapter, edit, guards, cfg, calibration_data=cal)
        pm = report.metrics.get("primary_metric", {})
        assert pm.get("final") == 1.0 and pm.get("preview") == 1.0
    finally:
        del os.environ["INVARLOCK_DEBUG_TRACE"]


def test_compute_real_metrics_config_none(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    model = Toy()
    adapter = DummyAdapter()
    metrics, _ = runner._compute_real_metrics(
        model,
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ],
        adapter,
        preview_n=1,
        final_n=1,
        config=None,
    )
    pm = metrics.get("primary_metric", {})
    assert isinstance(pm, dict) and pm.get("final") and pm.get("preview")


def test_bootstrap_replicates_disable(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    model = Toy()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("eval", {}).update(
        {"bootstrap": {"enabled": True, "replicates": 0}}
    )
    metrics, _ = runner._compute_real_metrics(
        model,
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ],
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )
    pm = metrics.get("primary_metric", {})
    assert isinstance(pm, dict) and pm.get("final") and pm.get("preview")


def test_bootstrap_unknown_method_disabled(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    model = Toy()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("eval", {}).update(
        {"bootstrap": {"enabled": True, "method": "weird", "replicates": 0}}
    )
    metrics, _ = runner._compute_real_metrics(
        model,
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ],
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )
    pm = metrics.get("primary_metric", {})
    assert isinstance(pm, dict) and pm.get("final") and pm.get("preview")


def test_preview_zero_final_positive_path(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    model = Toy()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    metrics, _ = runner._compute_real_metrics(
        model,
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ],
        adapter,
        preview_n=0,
        final_n=1,
        config=cfg,
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") > 0


def test_bootstrap_dataset_seed_str(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3, bias=False)

        def forward(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    runner = CoreRunner()
    model = Toy()
    adapter = DummyAdapter()
    cfg = make_config(tmp_path)
    cfg.context.setdefault("eval", {}).update(
        {"bootstrap": {"enabled": True, "method": "percentile", "replicates": 2}}
    )
    cfg.context.setdefault("dataset", {})["seed"] = (
        "abc"  # triggers fallback seed parsing
    )
    metrics, _ = runner._compute_real_metrics(
        model,
        [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}],
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )
    pm = metrics.get("primary_metric", {})
    assert pm.get("final") and pm.get("preview")


def test_measure_latency_tensor_inputs_variants():
    import torch

    runner = CoreRunner()

    class M:
        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    # 1-D tensor → unsqueeze path
    s1 = {"input_ids": torch.tensor([1, 2, 3])}
    # 2-D tensor → no unsqueeze
    s2 = {"input_ids": torch.tensor([[1, 2, 3]])}
    assert (
        runner._measure_latency(M(), [s1], "cpu") == 0.0
        or runner._measure_latency(M(), [s1], "cpu") > 0.0
    )
    assert (
        runner._measure_latency(M(), [s2], "cpu") == 0.0
        or runner._measure_latency(M(), [s2], "cpu") > 0.0
    )


def test_measure_latency_device_to_exception():
    runner = CoreRunner()

    class M:
        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Obj()

    # Nonstandard device object forces .to(device) to raise
    bad_device = object()
    sample = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    with pytest.raises(
        RuntimeError, match="Latency measurement device transfer failed"
    ):
        runner._measure_latency(M(), [sample], bad_device)


def test_runner_resolve_policies_error(monkeypatch, tmp_path):
    # Resolver failures should now surface instead of being normalized away.
    from invarlock.core import runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "resolve_tier_policies",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oops")),
    )
    from invarlock.core.api import RunReport

    r = CoreRunner()
    model = DummyModel()
    adapter = DummyAdapter()
    guards = [GoodGuard()]
    cfg = make_config(tmp_path)
    # Call prepare guards phase directly to isolate the branch
    r._initialize_services(cfg)
    with pytest.raises(RuntimeError, match="oops"):
        r._prepare_guards_phase(
            model,
            adapter,
            guards,
            calibration_data=None,
            report=RunReport(),
            auto_config=None,
            config=cfg,
        )
    r._cleanup_services()
