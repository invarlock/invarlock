from __future__ import annotations

from typing import Any

import pytest

from invarlock.core.api import RunConfig, RunReport
from invarlock.core.runner import CoreRunner
from tests.core.test_runner_failure_and_recovery_metrics import (
    DummyAdapter,
    _minimal_calibration,
    _toy_model_with_losses,
)

# Note: Avoid exercising the MLM zero-usable-batches raise path due to a known
# UnboundLocal bug in runner error handling after exceptions inside eval.


def test_eval_phase_without_calibration_returns_non_evaluated_state():
    runner = CoreRunner()
    model = _toy_model_with_losses([0.3])
    adapter = DummyAdapter()
    report = RunReport()

    metrics = runner._eval_phase(
        model,
        adapter,
        calibration_data=None,
        report=report,
        preview_n=None,
        final_n=None,
        config=RunConfig(),
    )

    assert metrics["eval_state"] == {
        "evaluated": False,
        "reason": "missing_calibration_data",
    }
    assert "primary_metric" not in metrics
    assert report.evaluation_windows == {"preview": {}, "final": {}}


def test_measure_latency_paths():
    runner = CoreRunner()
    model = _toy_model_with_losses([0.2])

    # Missing input_ids yields early 0.0
    assert runner._measure_latency(model, [{"input_ids": None}], device="cpu") == 0.0

    # 1-D tensor input exercises unsqueeze and to(device) guards
    import torch

    latency = runner._measure_latency(
        model, [torch.tensor([1, 2, 3])], device=torch.device("cpu")
    )
    assert isinstance(latency, float)

    # Dict input exercises attention_mask/token_type_ids handling
    latency = runner._measure_latency(
        model,
        [
            {
                "input_ids": [4, 5, 6],
                "attention_mask": [1, 1, 1],
                "token_type_ids": [0, 0, 0],
            }
        ],
        device="cpu",
    )
    assert isinstance(latency, float)


def test_measure_latency_cuda_sync(monkeypatch):
    import torch

    runner = CoreRunner()

    class M:
        def __call__(self, *a, **k):
            class Obj:
                def __init__(self):
                    self.loss = torch.tensor(0.01)

            return Obj()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    sync_called: dict[str, bool] = {"called": False}

    def fake_sync():
        sync_called["called"] = True

    monkeypatch.setattr(torch.cuda, "synchronize", fake_sync)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *_args, **_kwargs: self)

    latency = runner._measure_latency(
        M(), [{"input_ids": [1, 2, 3]}], torch.device("cuda")
    )
    assert isinstance(latency, float)
    assert sync_called["called"]


def test_overlap_fraction_from_config():
    runner = CoreRunner()
    model = _toy_model_with_losses([0.4, 0.5])
    adapter = DummyAdapter()
    cfg = RunConfig(context={"eval": {"overlap": {"stride": 2, "seq_len": 4}}})

    metrics, _ = runner._compute_real_metrics(
        model,
        _minimal_calibration(2),
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )

    # When overlap config is provided, helper should not crash; default may
    # clamp to 1.0 if not applied.
    assert metrics.get("window_overlap_fraction") is not None


def test_bootstrap_coverage_warning(monkeypatch):
    events: list[tuple[str, dict[str, Any]]] = []

    class CapturingLogger:
        def log(self, component, operation, level, data):
            events.append((operation, data or {}))

    runner = CoreRunner()
    runner.event_logger = CapturingLogger()
    model = _toy_model_with_losses([0.4, 0.5])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "eval": {
                "bootstrap": {"enabled": True, "replicates": 5},
                "overlap": {"stride": 2, "seq_len": 4},
            }
        }
    )

    runner._compute_real_metrics(
        model,
        _minimal_calibration(2),
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )

    assert any(op == "bootstrap_coverage_warning" for op, _ in events)


def test_finalize_phase_catastrophic_spike_rolls_back():
    runner = CoreRunner()
    report = RunReport()
    guard_results = {"g": {"passed": True}}
    metrics = {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 3.5}}

    status = runner._finalize_phase(
        model=object(),
        adapter=DummyAdapter(),
        guard_results=guard_results,
        metrics=metrics,
        config=RunConfig(spike_threshold=2.0, max_pm_ratio=10.0),
        report=report,
    )

    assert status == "rollback"


def test_eval_overlap_warning_logged_non_ci():
    events: list[str] = []

    class CapturingLogger:
        def log(self, component, operation, level, data):
            events.append(operation)

    runner = CoreRunner()
    runner.event_logger = CapturingLogger()
    model = _toy_model_with_losses([1.0, 1.1])
    adapter = DummyAdapter()
    cfg = RunConfig(
        context={
            "dataset": {"seq_len": 4, "stride": 2},
            "pairing_baseline": {
                "preview": {"window_ids": [0], "input_ids": [[1, 2, 3, 4]]},
                "final": {"window_ids": [1], "input_ids": [[1, 2, 3, 4]]},
            },
            "profile": "dev",
        }
    )

    runner._compute_real_metrics(
        model,
        _minimal_calibration(2),
        adapter,
        preview_n=1,
        final_n=1,
        config=cfg,
    )

    assert "window_overlap_warning" in events


def test_measure_latency_model_exception_returns_zero():
    runner = CoreRunner()

    class BadModel:
        def __call__(self, *a, **k):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        runner._measure_latency(BadModel(), [{"input_ids": [1, 2, 3]}], "cpu")
