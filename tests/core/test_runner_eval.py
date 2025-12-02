import math
from unittest.mock import Mock

import torch
import torch.nn as nn

from invarlock.core import runner as core_runner
from invarlock.core.api import ModelAdapter, RunConfig, RunReport
from invarlock.core.runner import CoreRunner
from invarlock.core.types import RunStatus


class DummyOutput:
    def __init__(self, loss: float) -> None:
        self.loss = torch.tensor(loss, dtype=torch.float32)


class DummyModel(nn.Module):
    def __init__(self, losses: list[float]):
        super().__init__()
        self.losses = losses
        self.pointer = 0
        self.weight = nn.Parameter(torch.zeros(1))  # ensures parameters exist

    def forward(self, input_ids, attention_mask=None, labels=None):  # noqa: D401
        loss = self.losses[self.pointer % len(self.losses)]
        self.pointer += 1
        return DummyOutput(loss)


def _make_calibration(batch_count: int, seq_len: int = 8):
    batches = []
    for _ in range(batch_count):
        tensor = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
        batches.append({"input_ids": tensor})
    return batches


def test_core_runner_emits_bootstrap_confidence_intervals():
    preview_losses = [3.0, 3.1, 2.9, 3.05]
    final_losses = [3.3, 3.2, 3.4, 3.25]
    model = DummyModel(preview_losses + final_losses)
    calibration = _make_calibration(8)

    runner = CoreRunner()
    adapter = Mock(spec=ModelAdapter)

    config = RunConfig(
        context={
            "eval": {
                "bootstrap": {
                    "enabled": True,
                    "method": "bca_paired_delta_log",
                    "replicates": 128,
                    "alpha": 0.05,
                    "seed": 123,
                }
            },
            "dataset": {"seed": 42},
        }
    )

    metrics, eval_windows = runner._compute_real_metrics(
        model,
        calibration,
        adapter,
        preview_n=4,
        final_n=4,
        config=config,
    )

    # Log-space confidence intervals available for preview/final and delta
    log_preview_ci = metrics["logloss_preview_ci"]
    log_final_ci = metrics["logloss_final_ci"]
    log_delta_ci = metrics["logloss_delta_ci"]

    assert isinstance(log_preview_ci, tuple) and len(log_preview_ci) == 2
    assert isinstance(log_final_ci, tuple) and len(log_final_ci) == 2
    assert isinstance(log_delta_ci, tuple) and len(log_delta_ci) == 2

    # Confidence intervals should be finite and non-degenerate
    assert all(torch.isfinite(torch.tensor(log_preview_ci)))
    assert all(torch.isfinite(torch.tensor(log_final_ci)))
    assert all(torch.isfinite(torch.tensor(log_delta_ci)))

    assert abs(log_preview_ci[1] - log_preview_ci[0]) > 1e-6
    assert abs(log_final_ci[1] - log_final_ci[0]) > 1e-6
    assert abs(log_delta_ci[1] - log_delta_ci[0]) > 1e-6

    # Ratio CI bounds (exp of delta log CI) should be positive
    ratio_ci = (math.exp(log_delta_ci[0]), math.exp(log_delta_ci[1]))
    assert ratio_ci[0] > 0 and ratio_ci[1] > 0

    assert "final" in eval_windows and "preview" in eval_windows
    assert "logloss_preview" in metrics
    assert "logloss_final" in metrics
    assert metrics["bootstrap"]["method"] == "bca_paired_delta_log"
    assert metrics["bootstrap"]["replicates"] >= 128
    assert 0.0 <= metrics["window_overlap_fraction"] <= 1.0
    assert 0.0 <= metrics["window_match_fraction"] <= 1.0
    coverage = metrics["bootstrap"].get("coverage", {})
    assert coverage.get("preview", {}).get("used", 0) >= 1
    assert coverage.get("final", {}).get("used", 0) >= 1
    assert "window_duplicate_fraction" in metrics["bootstrap"]
    assert "window_match_fraction" in metrics["bootstrap"]
    assert len(eval_windows["preview"]["logloss"]) == 4
    assert len(eval_windows["final"]["logloss"]) == 4


def test_eval_windows_are_disjoint_and_paired_delta_summary():
    preview_losses = [3.0, 3.1, 3.2, 3.05]
    final_losses = [3.4, 3.3, 3.25, 3.5]
    model = DummyModel(preview_losses + final_losses)
    calibration = _make_calibration(8)

    runner = CoreRunner()
    adapter = Mock(spec=ModelAdapter)
    config = RunConfig(
        context={
            "eval": {
                "bootstrap": {
                    "enabled": True,
                    "method": "bca_paired_delta_log",
                    "replicates": 128,
                    "alpha": 0.05,
                    "seed": 321,
                }
            },
            "dataset": {"seed": 99},
        }
    )

    metrics, eval_windows = runner._compute_real_metrics(
        model,
        calibration,
        adapter,
        preview_n=4,
        final_n=4,
        config=config,
    )

    preview_ids = eval_windows["preview"]["window_ids"]
    final_ids = eval_windows["final"]["window_ids"]
    assert preview_ids == [0, 1, 2, 3]
    assert final_ids == [4, 5, 6, 7]

    assert metrics["paired_windows"] == 4
    summary = metrics["paired_delta_summary"]
    assert abs(summary["mean"]) > 1e-6
    assert summary["std"] >= 0.0


def test_finalize_phase_catastrophic_spike_triggers_guard_recovery():
    runner = CoreRunner()
    report = RunReport()
    report.meta["initial_checkpoint"] = "ckpt-initial"

    class DummyCheckpointManager:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object, str]] = []

        def restore_checkpoint(self, model, adapter, checkpoint_id) -> None:
            self.calls.append((model, adapter, checkpoint_id))

    runner.checkpoint_manager = DummyCheckpointManager()

    guard_results = {"variance": {"passed": True}}
    metrics = {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 2.75}}
    config = RunConfig()
    config.max_pm_ratio = 10.0  # Ensure catastrophic check is the driver

    model = object()
    adapter = object()
    status = runner._finalize_phase(
        model,
        adapter,
        guard_results,
        metrics,
        config,
        report,
    )

    assert status == RunStatus.ROLLBACK.value
    assert report.meta["guard_recovered"] is True
    assert report.meta["rollback_checkpoint"] == "ckpt-initial"
    assert "catastrophic_ppl_spike" in report.meta["rollback_reason"]
    assert runner.checkpoint_manager.calls == [(model, adapter, "ckpt-initial")]


def test_collect_cuda_flags_includes_workspace_env(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    flags = core_runner._collect_cuda_flags()
    assert flags["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
