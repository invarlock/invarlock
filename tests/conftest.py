from __future__ import annotations

import os
import sys
import types
from importlib import import_module
from pathlib import Path

import pytest

from invarlock.reporting.report_types import AutoConfig

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


def install_transformers_tokenizer_stub() -> None:
    """Install a tiny transformers tokenizer stub for import-only CLI tests."""
    try:
        import_module("transformers")
        return
    except (ImportError, ModuleNotFoundError):
        pass

    if "transformers" not in sys.modules:
        tr = types.ModuleType("transformers")

        class _Tok:
            pad_token = "<pad>"
            eos_token = "<eos>"

            def get_vocab(self) -> dict[str, int]:
                return {"<pad>": 0, "<eos>": 1}

        class _Auto:
            @staticmethod
            def from_pretrained(*_args: object, **_kwargs: object) -> _Tok:
                return _Tok()

        class _GPT2(_Auto):
            pass

        tr.AutoTokenizer = _Auto
        tr.GPT2Tokenizer = _GPT2
        sys.modules["transformers"] = tr

    if "transformers.tokenization_utils_base" not in sys.modules:
        sub = types.ModuleType("transformers.tokenization_utils_base")
        sub.PreTrainedTokenizerBase = object
        sys.modules["transformers.tokenization_utils_base"] = sub


def make_test_auto_config(
    *,
    enabled: bool = False,
    tier: str = "balanced",
    probes_used: int = 0,
    target_pm_ratio: float | None = None,
) -> AutoConfig:
    """Return a fully typed AutoConfig for report fixture builders."""
    return AutoConfig(
        enabled=enabled,
        tier=tier,
        probes_used=probes_used,
        target_pm_ratio=target_pm_ratio,
    )


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
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _VALID_TEST_IMAGE_DIGEST)
    yield


@pytest.fixture(autouse=True)
def _path_write_text_with_append(monkeypatch: pytest.MonkeyPatch):
    # Some tests use Path.write_text(..., append=True) which is not available
    # in all Python versions. Provide a local append-capable test helper.
    def _write_text(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        *,
        append: bool = False,
    ) -> int:
        if append:
            # Append mode
            self.parent.mkdir(parents=True, exist_ok=True)
            with self.open(
                "a", encoding=encoding, errors=errors, newline=newline
            ) as fh:
                return fh.write(data)
        with self.open("w", encoding=encoding, errors=errors, newline=newline) as fh:
            return fh.write(data)

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
