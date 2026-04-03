from __future__ import annotations

from invarlock.core.run_orchestrator_execute import _coerce_float, _coerce_int


def test_run_orchestrator_execute_coercers_handle_invalid_and_nonfinite_values() -> None:
    assert _coerce_float("3.25", default=1.0) == 3.25
    assert _coerce_float(float("nan"), default=1.0) == 1.0
    assert _coerce_float(float("inf"), default=1.0) == 1.0
    assert _coerce_float("bad", default=1.0) == 1.0

    assert _coerce_int("7", default=2) == 7
    assert _coerce_int("bad", default=2) == 2
