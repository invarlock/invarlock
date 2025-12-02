from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _reset_plugin_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Normalize plugin-discovery env for CLI tests.

    CI sets INVARLOCK_DISABLE_PLUGIN_DISCOVERY=1; most CLI plugin tests expect
    discovery to be enabled unless they explicitly set it, so default to "0" here.
    Tests that need discovery disabled still set the env per invocation.
    """
    monkeypatch.setenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY", "0")
    yield
