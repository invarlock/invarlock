from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from invarlock.core.api import ModelAdapter, RunConfig, RunReport
from invarlock.core.runner import CoreRunner
from invarlock.core.types import RunStatus


def test_finalize_rollback_on_guards_failed():
    runner = CoreRunner()
    report = RunReport()
    metrics = {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}}
    guard_results = {"spectral": {"passed": False}}
    cfg = RunConfig(max_pm_ratio=2.0)

    status = runner._finalize_phase(
        object(), object(), guard_results, metrics, cfg, report
    )
    assert status == RunStatus.ROLLBACK.value


def test_finalize_rollback_on_metrics_unacceptable():
    runner = CoreRunner()
    report = RunReport()
    metrics = {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 3.0}}
    guard_results = {"spectral": {"passed": True}}
    cfg = RunConfig(max_pm_ratio=1.5)

    status = runner._finalize_phase(
        object(), object(), guard_results, metrics, cfg, report
    )
    assert status == RunStatus.ROLLBACK.value


def test_finalize_success_when_all_good():
    runner = CoreRunner()
    report = RunReport()
    metrics = {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.02}}
    guard_results = {"spectral": {"passed": True}, "variance": {"passed": True}}
    cfg = RunConfig(max_pm_ratio=1.5)

    status = runner._finalize_phase(
        object(), object(), guard_results, metrics, cfg, report
    )
    assert status == RunStatus.SUCCESS.value


def test_finalize_skips_ratio_gate_for_non_ppl_metric():
    runner = CoreRunner()
    report = RunReport()
    metrics = {"primary_metric": {"kind": "accuracy", "final": 0.9}}
    guard_results = {"spectral": {"passed": True}}
    cfg = RunConfig(max_pm_ratio=1.01, spike_threshold=1.05)

    status = runner._finalize_phase(
        object(), object(), guard_results, metrics, cfg, report
    )
    assert status == RunStatus.SUCCESS.value


def test_finalize_skips_ratio_gate_when_preview_missing():
    runner = CoreRunner()
    report = RunReport()
    metrics = {"primary_metric": {"kind": "ppl_causal", "preview": 0.0, "final": 2.0}}
    guard_results = {"spectral": {"passed": True}}
    cfg = RunConfig(max_pm_ratio=1.01, spike_threshold=1.05)

    status = runner._finalize_phase(
        object(), object(), guard_results, metrics, cfg, report
    )
    assert status == RunStatus.SUCCESS.value


class DummyOutput:
    def __init__(self, loss: float) -> None:
        self.loss = torch.tensor(loss, dtype=torch.float32)


class DummyModel(nn.Module):
    def __init__(self, losses: list[float]):
        super().__init__()
        self.losses = losses
        self.ptr = 0
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, labels=None):  # noqa: D401
        loss = self.losses[self.ptr % len(self.losses)]
        self.ptr += 1
        return DummyOutput(loss)


def _make_calibration(batch_count: int, seq_len: int = 8):
    batches = []
    for _ in range(batch_count):
        tensor = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
        batches.append({"input_ids": tensor})
    return batches


def test_compute_real_metrics_degenerate_delta_marks_flag_in_ci():
    model = DummyModel([3.0, 3.2])
    calibration = _make_calibration(2)
    runner = CoreRunner()
    adapter = Mock(spec=ModelAdapter)

    cfg = RunConfig(context={"eval": {"bootstrap": {"enabled": True}}, "profile": "ci"})

    metrics, _ = runner._compute_real_metrics(
        model, calibration, adapter, preview_n=1, final_n=1, config=cfg
    )
    summary = metrics["paired_delta_summary"]
    assert summary["degenerate"] is True
    assert summary["degenerate_reason"] in {"no_pairs", "single_pair", "no_variation"}


def test_finalize_rollback_with_checkpoint_restore_called():
    runner = CoreRunner()
    report = RunReport()
    report.meta["initial_checkpoint"] = "cp-1"
    metrics = {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 3.0}}
    guard_results = {"spectral": {"passed": False}}
    cfg = RunConfig(max_pm_ratio=1.5)

    called = {}

    class StubCM:
        def restore_checkpoint(self, model, adapter, checkpoint_id):
            called["id"] = checkpoint_id
            return True

    runner.checkpoint_manager = StubCM()

    status = runner._finalize_phase(
        object(), object(), guard_results, metrics, cfg, report
    )
    assert status == RunStatus.ROLLBACK.value and called.get("id") == "cp-1"


def test_finalize_phase_records_rollback_failed_when_restore_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CoreRunner()
    report = RunReport()
    report.meta["initial_checkpoint"] = "cp-1"
    metrics = {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 3.0}}
    guard_results = {"spectral": {"passed": False}}
    cfg = RunConfig(max_pm_ratio=1.5)

    events: list[tuple[str, str, dict | None]] = []

    def patched_log(component, operation, level, data=None):  # type: ignore[no-untyped-def]
        events.append((component, operation, data))

    monkeypatch.setattr(CoreRunner, "_log_event", staticmethod(patched_log))

    class StubCM:
        def restore_checkpoint(self, model, adapter, checkpoint_id):  # type: ignore[no-untyped-def]
            return False

    runner.checkpoint_manager = StubCM()

    status = runner._finalize_phase(
        object(), object(), guard_results, metrics, cfg, report
    )
    assert status == RunStatus.ROLLBACK.value
    assert report.meta.get("rollback_failed") is True
    assert any(op == "rollback_failed" for _, op, _ in events)
