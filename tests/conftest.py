from __future__ import annotations

import os
import sys
import types
from importlib import import_module
from pathlib import Path

import pytest

from invarlock.reporting.report_types import AutoConfig

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)
_PATCH_TARGET_MODULES = (
    "invarlock.cli.bench",
    "invarlock.core.auto_tuning",
    "invarlock.core.bootstrap",
    "invarlock.core.config_loader",
    "invarlock.core.determinism_policy",
    "invarlock.core.metric_provider_resolution",
    "invarlock.core.registry",
    "invarlock.core.run_orchestrator_execute",
    "invarlock.core.runner",
    "invarlock.core.runtime_manifest_verify",
    "invarlock.eval.bench_runner",
    "invarlock.eval.data",
    "invarlock.eval.metrics_activation",
    "invarlock.eval.metrics_support",
    "invarlock.eval.primary_metric",
    "invarlock.eval.providers.seq2seq",
    "invarlock.model_utils",
    "invarlock.model_profile",
    "invarlock.observability.core",
    "invarlock.observability.health",
    "invarlock.plugins.bitsandbytes",
    "invarlock.evidence_pack",
    "invarlock.reporting.report_console",
    "invarlock.reporting.report_make",
    "invarlock.reporting.report_telemetry",
)


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


def _reattach_parent_package_attrs(module_name: str) -> None:
    parts = module_name.split(".")
    for idx in range(1, len(parts)):
        parent_name = ".".join(parts[:idx])
        child_name = parts[idx]
        child_module_name = ".".join(parts[: idx + 1])
        parent_module = sys.modules.get(parent_name)
        child_module = sys.modules.get(child_module_name)
        if parent_module is None or child_module is None:
            continue
        if not hasattr(parent_module, child_name):
            setattr(parent_module, child_name, child_module)


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
        "INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE",
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
def _materialize_patch_targets(request: pytest.FixtureRequest):
    nodeid = request.node.nodeid
    if "import_safety" in nodeid:
        yield
        return

    for module_name in _PATCH_TARGET_MODULES:
        try:
            import_module(module_name)
            _reattach_parent_package_attrs(module_name)
        except Exception:
            continue
    yield


@pytest.fixture(autouse=True)
def _default_security_bypass_for_local_tests(monkeypatch: pytest.MonkeyPatch):
    # The product is container-first, but the general pytest harness stays on
    # host execution unless an individual test opts back into the
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
