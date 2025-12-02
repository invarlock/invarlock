import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class FailingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Provide at least one parameter so device resolution works
        self._stub = nn.Linear(1, 1)

    def forward(self, input_ids, labels=None):  # type: ignore[override]
        raise RuntimeError("intentional failure")


def test_calibration_insufficient_coverage_sets_reason_and_warns():
    model = FailingModel()
    guard = VarianceGuard(
        {
            "mode": "ci",
            "alpha": 0.05,
            "calibration": {"windows": 1, "min_coverage": 2, "seed": 123},
        }
    )

    # One calibration batch but model always fails → coverage stays 0 < min_coverage
    calibration_batches = [{"input_ids": torch.zeros(2, 4)}]

    # Call internal evaluation helper directly to precisely hit the branch
    guard._evaluate_calibration_pass(
        model=model,
        calibration_batches=calibration_batches,
        min_coverage=2,
        calib_seed=123,
        tag="t",
    )

    state = getattr(guard, "_predictive_gate_state", {})
    assert state.get("reason") == "insufficient_coverage"

    # Event log should include prepare_monitor_mode warning
    operations = [e.get("operation") for e in guard.events]
    assert "prepare_monitor_mode" in operations
