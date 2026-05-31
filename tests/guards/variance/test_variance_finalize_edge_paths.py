import torch

from invarlock.guards.variance import VarianceGuard


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


def test_variance_finalize_mismatch_and_warnings(monkeypatch):
    g = VarianceGuard()
    # Pretend prepare() already happened
    g._prepared = True
    g._post_edit_evaluated = True

    # Seed/tap/policy context
    g._policy["seed"] = 777
    g._policy["min_gain"] = 0.01
    g._policy["tie_breaker_deadband"] = 0.02

    # Set A/B outcomes and context
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 99.98  # Below absolute floor improvement (0.02 < 0.05)
    g._ab_gain = 0.01  # Below min_gain + deadband when enabled
    g._ab_windows_used = 3
    g._ab_seed_used = 123  # different from policy seed -> warning

    # Force predictive gate evaluated and passed
    g._predictive_gate_state = {
        "evaluated": True,
        "passed": True,
        "reason": "ci_gain_met",
    }

    # Simulate prior attempts and leftover checkpoints
    g._enable_attempt_count = 4
    g._disable_attempt_count = 5
    g._checkpoint_stack = [object()]

    # Start disabled; A/B gate will say to enable
    g._enabled = False

    # Force AB gate decision via monkeypatch
    monkeypatch.setattr(g, "_evaluate_ab_gate", lambda: (True, "criteria_met"))

    # Avoid touching real modules; stub enable/disable
    monkeypatch.setattr(g, "enable", lambda model: True)
    monkeypatch.setattr(g, "disable", lambda model: True)

    out = g.finalize(DummyModel())

    # Expect failures due to deadband and absolute floor not met
    assert out["passed"] is False
    msg = "VE enabled without meeting tie-breaker deadband"
    assert any(msg in e for e in out["errors"])  # deadband error
    msg2 = "VE enabled without meeting absolute floor"
    assert any(msg2 in e for e in out["errors"])  # absolute floor error

    # Warnings should include seed mismatch, attempts and uncommitted checkpoints
    warns = "\n".join(out.get("warnings", []))
    assert "unexpected seed" in warns
    assert "Multiple enable attempts" in warns
    assert "Multiple disable attempts" in warns
    assert "Uncommitted checkpoints remaining" in warns
