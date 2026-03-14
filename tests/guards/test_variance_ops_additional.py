from __future__ import annotations

from types import SimpleNamespace

import torch

from invarlock.guards.variance_ops import disable_guard, pop_checkpoint, push_checkpoint


class _Module:
    def __init__(self, weight=None):
        if weight is not None:
            self.weight = weight


def _guard() -> SimpleNamespace:
    events: list[tuple[str, dict[str, object]]] = []
    return SimpleNamespace(
        _target_modules={},
        _checkpoint_stack=[],
        _log_event=lambda operation, **data: events.append((operation, data)),
        _events=events,
        _enabled=True,
        _disable_attempt_count=0,
        _scales={},
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


def test_disable_guard_falls_back_when_checkpoint_restore_fails(monkeypatch) -> None:
    guard = _guard()
    module = _Module(torch.ones((2, 2)))
    guard._checkpoint_stack = [{"layer": module.weight.clone()}]
    guard._target_modules = {"layer": module}
    guard._scales = {"layer": 2.0}

    monkeypatch.setattr(
        "invarlock.guards.variance_ops.pop_checkpoint",
        lambda *_args, **_kwargs: False,
    )

    assert disable_guard(guard, model=None) is True
    assert guard._enabled is False
    assert any(event[0] == "disable_checkpoint_failed" for event in guard._events)
