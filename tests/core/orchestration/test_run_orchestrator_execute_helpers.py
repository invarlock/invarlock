from __future__ import annotations

from invarlock.core.orchestration.execute import (
    _cleanup_snapshot_tmpdir,
    _coerce_float,
    _coerce_int,
)
from invarlock.core.run_orchestrator import RunCleanupStatusEvent


def test_run_orchestrator_execute_coercers_handle_invalid_and_nonfinite_values() -> (
    None
):
    assert _coerce_float("3.25", default=1.0) == 3.25
    assert _coerce_float(float("nan"), default=1.0) == 1.0
    assert _coerce_float(float("inf"), default=1.0) == 1.0
    assert _coerce_float("bad", default=1.0) == 1.0

    assert _coerce_int("7", default=2) == 7
    assert _coerce_int(float("inf"), default=2) == 2
    assert _coerce_int("bad", default=2) == 2


def test_cleanup_snapshot_tmpdir_swallows_emit_failures() -> None:
    events: list[object] = []

    def _emit(event: object) -> None:
        events.append(event)
        raise TypeError("sink unavailable")

    result = _cleanup_snapshot_tmpdir(
        snapshot_tmpdir="unused",
        no_cleanup=True,
        emit=_emit,
    )
    assert result is None
    assert len(events) == 1
    assert isinstance(events[0], RunCleanupStatusEvent)
    assert events[0].removed is False
