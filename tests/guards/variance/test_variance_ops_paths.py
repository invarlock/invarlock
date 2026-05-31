from __future__ import annotations

from types import SimpleNamespace

import torch

import invarlock.guards as guards_pkg
import invarlock.guards.variance_ops as variance_ops
from invarlock.guards.variance_ops import (
    commit_checkpoint,
    disable_guard,
    enable_guard,
    pop_checkpoint,
    push_checkpoint,
)

guards_pkg.variance_ops = variance_ops


class _Module:
    def __init__(self, weight=None):
        if weight is not None:
            self.weight = weight


class _MPSWeight:
    def __init__(self, value: float) -> None:
        self.data = torch.full((2, 2), value)
        self.dtype = self.data.dtype

    @property
    def device(self) -> str:
        return "mps:0"


def _guard() -> SimpleNamespace:
    events: list[tuple[str, dict[str, object]]] = []
    return SimpleNamespace(
        _target_modules={},
        _checkpoint_stack=[],
        _log_event=lambda operation, **data: events.append((operation, data)),
        _events=events,
        _enabled=True,
        _disable_attempt_count=0,
        _enable_attempt_count=0,
        _scales={},
        _original_scales={},
        _prepared=True,
        _monitor_only=False,
        _scale_matches_target=lambda scale_name, target_name: scale_name == target_name,
    )


def test_push_checkpoint_skips_modules_without_weight() -> None:
    guard = _guard()
    guard._target_modules = {"noweight": _Module()}

    push_checkpoint(guard, model=None)

    assert len(guard._checkpoint_stack) == 1
    assert guard._checkpoint_stack[0] == {}


def test_pop_checkpoint_ignores_missing_targets_and_missing_weight() -> None:
    guard = _guard()
    weight = torch.ones((2, 2))
    guard._checkpoint_stack = [{"missing": weight.clone(), "noweight": weight.clone()}]
    guard._target_modules = {"noweight": _Module()}

    assert pop_checkpoint(guard, model=None) is True
    assert guard._checkpoint_stack == []


def test_commit_checkpoint_pops_latest_snapshot_and_logs() -> None:
    guard = _guard()
    guard._checkpoint_stack = [
        {"first": torch.ones((1, 1))},
        {"second": torch.zeros((1, 1))},
    ]

    commit_checkpoint(guard)

    assert len(guard._checkpoint_stack) == 1
    assert "first" in guard._checkpoint_stack[0]
    assert any(event[0] == "checkpoint_committed" for event in guard._events)


def test_disable_guard_falls_back_when_checkpoint_restore_fails(monkeypatch) -> None:
    guard = _guard()
    module = _Module(torch.ones((2, 2)))
    guard._checkpoint_stack = [{"layer": module.weight.clone()}]
    guard._target_modules = {"layer": module}
    guard._scales = {"layer": 2.0}

    monkeypatch.setattr(
        variance_ops,
        "pop_checkpoint",
        lambda *_args, **_kwargs: False,
    )

    assert disable_guard(guard, model=None) is True
    assert guard._enabled is False
    assert any(event[0] == "disable_checkpoint_failed" for event in guard._events)


def test_enable_guard_uses_mps_scaling_path() -> None:
    guard = _guard()
    weight = _MPSWeight(1.0)
    guard._enabled = False
    guard._target_modules = {"layer": _Module(weight)}
    guard._scales = {"layer": 2.0}

    assert enable_guard(guard, model=None) is True
    assert torch.allclose(weight.data, torch.full((2, 2), 2.0))


def test_enable_guard_logs_catastrophic_failure_when_commit_raises(
    monkeypatch,
) -> None:
    guard = _guard()
    guard._enabled = False
    guard._target_modules = {"layer": _Module(torch.ones((2, 2)))}
    guard._scales = {"layer": 1.5}

    monkeypatch.setattr(
        variance_ops,
        "commit_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert enable_guard(guard, model=None) is False
    assert any(event[0] == "enable_catastrophic_failure" for event in guard._events)


def test_disable_guard_uses_mps_revert_path() -> None:
    guard = _guard()
    weight = _MPSWeight(2.0)
    guard._target_modules = {"layer": _Module(weight)}
    guard._scales = {"layer": 2.0}

    assert disable_guard(guard, model=None) is True
    assert torch.allclose(weight.data, torch.full((2, 2), 1.0))


def test_disable_guard_logs_catastrophic_failure_when_completion_raises() -> None:
    guard = _guard()
    events: list[tuple[str, dict[str, object]]] = []

    def _log_event(operation: str, **data: object) -> None:
        if operation == "disable_complete":
            raise RuntimeError("boom")
        events.append((operation, data))

    guard._events = events
    guard._log_event = _log_event
    guard._target_modules = {"layer": _Module(torch.ones((2, 2)))}
    guard._scales = {"layer": 2.0}

    assert disable_guard(guard, model=None) is False
    assert any(operation == "disable_catastrophic_failure" for operation, _ in events)


def test_disable_guard_returns_false_when_zero_scale_cannot_be_reverted() -> None:
    guard = _guard()
    module = _Module(torch.ones((2, 2)))
    guard._target_modules = {"layer": module}
    guard._scales = {"layer": 0.0}

    assert disable_guard(guard, model=None) is False
    assert guard._enabled is True
    assert any(event[0] == "scale_revert_error" for event in guard._events)
    assert any(event[0] == "disable_failed" for event in guard._events)


def test_disable_guard_succeeds_with_empty_scales_when_enabled() -> None:
    guard = _guard()

    assert disable_guard(guard, model=None) is True
    assert guard._enabled is False
    assert any(event[0] == "disable_complete" for event in guard._events)
