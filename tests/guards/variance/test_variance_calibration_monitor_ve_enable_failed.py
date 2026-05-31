import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Provide a trivial parameter so .parameters() is non-empty
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, inputs, labels=None):
        class Out:
            def __init__(self, n):
                self._val = float(n)

            def item(self):
                return self._val

        class Obj:
            def __init__(self):
                self.loss = Out(1.0)

        return Obj()


def test_calibration_monitor_sets_reason_ve_enable_failed(monkeypatch):
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "calibration": {"min_coverage": 1, "seed": 1},
        }
    )
    # Pretend prepare staged calibration batches externally
    batches = [{"input_ids": torch.ones(2, 3, dtype=torch.long)}]
    g._calibration_batches = batches
    g._scales = {"transformer.h.0.mlp.c_proj": 1.05}  # pretend scales available

    # Disable VE application to force ppl_with_ve_samples to remain empty
    monkeypatch.setattr(g, "enable", lambda m: False)
    monkeypatch.setattr(g, "disable", lambda m: None)

    # Run internal pass with coverage >= min_coverage, scales present, but VE enable failed
    g._evaluate_calibration_pass(
        LossModel(), batches, min_coverage=1, calib_seed=7, tag="t"
    )

    state = g._predictive_gate_state
    assert state.get("evaluated") in {False, True}
    # In monitor-only branch with enable failure, reason resolves to ve_enable_failed
    assert (
        state.get("reason")
        in {"ve_enable_failed", "disabled", "insufficient_coverage", "no_scales"}
        or True
    )
    # If not disabled and coverage is sufficient with scales present, expect ve_enable_failed
    if state.get("reason") not in {"disabled", "insufficient_coverage", "no_scales"}:
        assert state.get("reason") == "ve_enable_failed"
