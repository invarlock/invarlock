from __future__ import annotations

import torch.nn as nn

from invarlock.core.run_report_payload_policy import build_guard_entries
from invarlock.guards.rmt import RMTGuard
from invarlock.guards.variance import VarianceGuard


def test_rmt_activation_required_failure_emits_strict_blocking_shape() -> None:
    guard = RMTGuard()
    model = nn.Linear(2, 2)

    guard.prepare(model, calib=None, policy={"activation_required": True})
    result = guard.validate(model, adapter=None, context={})

    assert result["supported"] is False
    assert result["reason"] == "activation_required"
    assert result["assurance_blocking"] is True

    [entry] = build_guard_entries({"rmt": result})
    assert entry["supported"] is False
    assert entry["reason"] == "activation_required"
    assert entry["assurance_blocking"] is True


def test_variance_unprepared_no_targets_emits_strict_blocking_shape() -> None:
    guard = VarianceGuard()

    result = guard.validate(nn.Linear(2, 2), adapter=None, context={})

    assert result["supported"] is False
    assert result["reason"] == "no_variance_targets"
    assert result["assurance_blocking"] is True

    [entry] = build_guard_entries({"variance": result})
    assert entry["supported"] is False
    assert entry["reason"] == "no_variance_targets"
    assert entry["assurance_blocking"] is True
