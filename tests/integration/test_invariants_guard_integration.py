import pytest
import torch

from invarlock.guards.invariants import InvariantsGuard


@pytest.mark.integration
def test_invariants_guard_detects_parameter_count_change():
    guard = InvariantsGuard(on_fail="abort")

    baseline_model = torch.nn.Linear(4, 4)
    prepare_info = guard.prepare(baseline_model, adapter=None, calib=None, policy={})
    assert prepare_info["ready"] is True

    mutated_model = torch.nn.Linear(8, 4)
    outcome = guard.finalize(mutated_model)

    assert outcome.passed is False
    assert outcome.action == "abort"
    assert any(
        violation.get("type") == "invariant_violation"
        for violation in outcome.violations
    )
