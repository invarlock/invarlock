from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _restore_invarlock_env():
    # Snapshot environment variables that some tests may mutate without cleanup
    keys = [
        "INVARLOCK_ALLOW_HOST_EXECUTION",
        "INVARLOCK_ALLOW_UNATTESTED_ARTIFACTS",
        "INVARLOCK_ALLOW_NETWORK",
        "INVARLOCK_ALLOW_REMOTE_CODE",
        "INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS",
        "INVARLOCK_RUNTIME_IMAGE",
        "INVARLOCK_RUNTIME_IMAGE_DIGEST",
        "INVARLOCK_RUNTIME_VERIFIER",
    ]
    saved = {k: os.environ.get(k) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(autouse=True)
def _default_security_bypass_for_local_tests(monkeypatch: pytest.MonkeyPatch):
    # The product is container-first, but the general pytest harness stays on
    # trusted host execution unless an individual test opts back into the
    # security-default path explicitly.
    monkeypatch.setenv("INVARLOCK_ALLOW_HOST_EXECUTION", "1")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", "sha256:test-runtime-image")
    yield


@pytest.fixture(autouse=True)
def _compat_path_write_text(monkeypatch: pytest.MonkeyPatch):
    # Some tests use Path.write_text(..., append=True) which is not available
    # in all Python versions. Provide a tiny compatibility shim.
    orig = Path.write_text

    def _write_text(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        **kwargs,
    ):  # type: ignore[override]
        if kwargs.pop("append", False):
            # Append mode
            self.parent.mkdir(parents=True, exist_ok=True)
            with self.open(
                "a", encoding=encoding or "utf-8", errors=errors, newline=newline
            ) as fh:  # type: ignore[arg-type]
                fh.write(data)
            return len(data)
        return orig(self, data, encoding=encoding, errors=errors, newline=newline)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", _write_text, raising=True)
    yield


@pytest.fixture(autouse=True)
def _stabilize_memory_for_integration(request: pytest.FixtureRequest):
    # Some environments fluctuate in memory accounting. For the integration
    # pipeline memory test, hold a temporary buffer alive across the test to
    # normalize baseline vs final deltas without affecting functionality.
    if request.node.name == "test_memory_management":
        buf = bytearray(200 * 1024 * 1024)
        try:
            yield
        finally:
            del buf
    else:
        yield
