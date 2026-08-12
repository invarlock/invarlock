from __future__ import annotations

import pytest

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


@pytest.fixture
def allow_host_execution_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INVARLOCK_ALLOW_HOST_EXECUTION", "1")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _VALID_TEST_IMAGE_DIGEST)
    yield
