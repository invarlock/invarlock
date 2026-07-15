from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import invarlock.guards.policies as policies_mod
from invarlock.guards.invariants import InvariantsGuard


def test_non_strict_invariants_context_preserves_configured_action() -> None:
    guard = InvariantsGuard(strict_mode=False, on_fail="rollback")
    guard.set_run_context(SimpleNamespace(context={"assurance": {"mode": "off"}}))

    assert guard.strict_mode is False
    assert guard.on_fail == "rollback"


def test_invariants_prepare_accepts_absent_optional_policy_without_escalation() -> None:
    guard = InvariantsGuard(strict_mode=False, on_fail="monitor")
    model = torch.nn.Linear(2, 2)
    result = guard.prepare(
        model,
        adapter=None,
        calib=None,
        policy=None,  # type: ignore[arg-type]
    )

    assert result["ready"] is True
    assert guard.prepared is True
    assert guard.strict_mode is False
    assert guard.on_fail == "monitor"
    assert guard.baseline_checks


def test_variance_policy_rejects_non_numeric_calibration_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        policies_mod,
        "get_tier_guard_config",
        lambda _tier, _guard: {
            "calibration": {"windows": True, "min_coverage": "bad", "seed": None}
        },
    )
    with pytest.raises(policies_mod.TierConfigError, match="packaged variance policy"):
        policies_mod.get_variance_policy("balanced")


def test_variance_policy_rejects_incomplete_calibration_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        policies_mod,
        "get_tier_guard_config",
        lambda _tier, _guard: {"calibration": {"windows": "invalid"}},
    )
    with pytest.raises(policies_mod.TierConfigError, match="packaged variance policy"):
        policies_mod.get_variance_policy("balanced")
