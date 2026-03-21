from __future__ import annotations

import pytest
import typer

from invarlock.cli import security_helpers


def test_maybe_delegate_model_command_delegates_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_HOST_EXECUTION", "0")
    monkeypatch.delenv("INVARLOCK_CONTAINER_EXECUTION", raising=False)
    calls: list[str] = []

    def _delegate() -> int:
        calls.append("delegated")
        return 0

    monkeypatch.setattr(
        security_helpers,
        "delegate_current_process_to_container",
        _delegate,
        raising=True,
    )

    with pytest.raises(typer.Exit) as exc:
        security_helpers.maybe_delegate_model_command()

    assert exc.value.exit_code == 0
    assert calls == ["delegated"]
